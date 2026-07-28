# E-LLM-1 postmortem — adversarial review of the P-G1 null and its diagnosis

*Methods review, 2026-07-27. All numbers computed CPU-only from on-disk
artifacts: teacher checkpoints (steps 25/50, both curriculum cells),
`gsm8k_partial/{teacher_vs_difficulty,cell_trajectories,ksweep_results}.json`,
and the lime-nlp/GSM8K_Difficulty cache (7470/7473 matched by
normalized-prefix on the question text). Monte-Carlo utilities use M=400
full-vector Thompson draws over the real 7473-prompt posterior; fix
simulations use a beta-binomial true-p field fitted to the run's own
1-visit outcome data. Methods in §A.*

## Verdict up front

The posterior-starvation diagnosis is **directionally correct but
incomplete, and the null is over-determined**. Three independent mechanisms
each cap the allocation channel near uniform, and a fourth caps what even a
perfect allocator could have bought:

1. **Starvation (the on-record diagnosis)** — 66% of the pool never
   visited; Beta(1,1) prior mass dominates the utility field. Real, and
   the largest single term. Confirmed.
2. **One visit buys almost nothing at this prior/utility shape** — a
   16/16-fail visit moves E[u(p̃)] only 0.44 → 0.41 (vs 0.64 live,
   0.78 peak). Even a zero-starvation run (every prompt visited once)
   yields only a ~1.5× live:dead sampling tilt.
3. **decay=0.7 caps the asymptote** — an always-failing prompt's posterior
   converges to Beta(1, 54.3), keeping E[u] ≈ 0.21 forever (~30% of peak).
   Max live:dead odds ≈ 3× at ANY run length. "Longer runs" climb toward
   this ceiling; they cannot pass it.
4. **The utility landscape itself is compressed at 360M** — the run's own
   1-visit data give mean p ≈ 0.04, 67% dead-per-visit: GSM8K@SmolLM2-360M
   sits between the regime map's "balanced" and "frontier-heavy" rows. The
   fitted per-prompt ORACLE collects only **1.88×** uniform's advantage
   mass (dead-sampled 0.69 → 0.40); the best oracle confined to the
   external-annotation axis collects **1.04×** (empirical, §2). Channel 1's
   entire headroom here is a factor ~2, and the axis-aligned part of it is
   a factor ~1.04.

Measured end state of the system as run: normalized ESS of the expected
sampling weights = **0.977** (1.0 = uniform); live-visited prompts weighted
1.35× uniform, dead-visited 0.91×, unvisited 0.96×; top-decile mass 0.141
vs 0.100. The teacher's realized policy was a **1.2–1.5× tilt toward the
~13% of the pool it had seen succeed** — real (directly visible in the
second-half visit split, §1e), and functionally indistinguishable from
uniform for a 50-step val outcome. P-G1's null follows structurally.

The floor hypothesis (a) is **refuted** as a cause; Thompson noise (d) is
secondary. For the paper: §7.5's "posterior-starved" sentence survives, but
should be stated as the two-bound version (starved posterior AND compressed
utility landscape), because the second bound is what survives every
visit-count fix and is what makes the E-LLM-2 tier design the right next
move *for Countdown specifically* (§2, Rank note).

---

## 1. The four hypotheses, quantified

Teacher checkpoint (maxrl_curtrue step 50): 7473 prompts, 2561 visited
(34.3%), 0.428 visits/prompt, 1.25 visits/visited. Visit histogram
{0:4912, 1:2001, 2:491, 3:60, 4:8, 5:1}. Of visited prompts, **1600
(62.5%) are dead** (α = 1.0 exactly — zero successes ever; matches the
~65% population dead-rate) and 961 live, with live p̂ mean 0.148 —
essentially *on* the utility peak p\* = 0.169. The grpo_curtrue teacher is
statistically identical (visited 2573, live 37%).

### (a) Does the 0.1 uniform floor dominate at n=7473? — NO

Expected sampling weight (MC over Thompson draws, floor included):

| class | E[weight] × uniform | floor share of weight |
|---|---|---|
| unvisited (n=4912) | 0.96× | 10% |
| dead-visited (n=1600) | 0.91× | 11% |
| live-visited (n=961) | **1.35×** | 7% |

The floor moves a dead prompt's weight from ~0.89× to 0.91× uniform —
negligible either way. The real problem is that the *non-floor* mass is
itself near-uniform: 62.9% of total utility mass sits on unvisited prompts
and 19.1% on dead-visited ones — **82% of the teacher's sampling mass goes
to prompts it has never seen succeed**. Floor changes (fix vi) are moot.

### (b) Does decay=0.7 reset the posterior between revisits? — NOT in this run; but it binds every "more visits" fix

Half-life = ln 0.5 / ln 0.7 = **1.94 observations** (effective memory
1/(1−0.7) = 3.33 obs ≈ 53 rollouts). At 1.25 visits/visited, decay never
engaged — so it did not cause this null. What it does is cap the payoff of
fixes that raise visit counts:

| all-fail visits | p̂ (decay .7) | u(p̂) | p̂ (no decay) | u(p̂) |
|---|---|---|---|---|
| 1 | .056 | .544 | .056 | .544 |
| 2 | .034 | .393 | .029 | .350 |
| 5 | .022 | .273 | .012 | .166 |
| ∞ | .018 | **.235** | →0 | →0 |

With decay 0.7, a permanently-dead prompt retains ≥30% of peak utility
forever (E[u(p̃)] at the Beta(1,54.3) fixed point = 0.21); a converged live
prompt (p=0.15) reaches E[u] = 0.75. Max odds ≈ 3×. The 0.7 default was
validated (V2b) where *tracking* dominates — many visits per arm. At <2
visits/prompt it only truncates evidence. Any re-run at low revisit rates
should use decay ≥ 0.9 or evidence-scaled decay; simulation (§2) confirms
decay is worth ~0 at 50 steps but matters by 150.

### (c) Can the utility differentiate LIVE prompts from each other? — NO

E[u(p̃)] under Thompson: unvisited **0.441**, dead-visited **0.412**,
live-visited **0.642** (peak 0.779). Two readings:

- One full 16-rollout failure moves E[u] only 0.44 → 0.41: Beta(1,17)
  keeps a fat upper tail, and u(p) is still 0.52 at the dead-visited
  posterior mean p̂ = 0.052. The utility is *designed* to stay high near
  the frontier — which at this pool composition is nearly everywhere.
- Live 1-visit posteriors are Beta(1+k, 17−k), k ∈ {1..4} for almost all;
  p̂ ∈ [0.11, 0.26] — a band on which u is a plateau (0.72–0.78). **The
  utility has no usable gradient across the live pool at 1-visit
  resolution.** The only decision expressible is the binary live/dead
  split, at 1.5× odds.

This is the precise sense in which the on-record diagnosis is incomplete:
it implies visit *count* is the binding constraint, but per-visit evidence
strength is a second, independent one. A zero-starvation counterfactual
(all 7473 prompts visited exactly once) still yields ESS ≈ 0.95.

### (d) Is Thompson noise the killer? — SECONDARY

Per-draw sd of u(p̃): 0.24 (unvisited/dead), 0.14 (live); class gap to
resolve 0.23 → per-draw SNR ≈ 1. P(dead out-draws live on one draw) =
0.22; P(unvisited out-draws live) = 0.26. This randomizes which prompt
within a class gets picked (hence the wobbly per-step dead-sampled
fraction, 0.48–0.75) but the mean allocation follows the E[u] ratios in
(c), which are the primary problem. γ > 1 (fix vi) would sharpen noise as
much as signal at these posterior widths.

### (e) Direct confirmation that the teacher acted — at exactly the predicted (tiny) magnitude

Second-half visit rates by step-25 class (visits₅₀ − visits₂₅ per prompt):

| cell | live@25 | dead@25 | unvisited@25 | realized live:dead tilt |
|---|---|---|---|---|
| maxrl_curtrue | 0.259 | 0.218 | 0.210 | **1.19×** |
| grpo_curtrue | 0.245 | 0.202 | 0.214 | **1.22×** |

The allocation channel *did* engage — by the ~1.2× the posterior math
predicts, invisible on the external difficulty axis: Spearman(visits,
difficulty) per half = −0.003/+0.002 (maxrl), −0.010/+0.003 (grpo); all
p > 0.4. Meanwhile the *evidence* channel is real in both cells
(Spearman(p̂, difficulty | visited) = −0.166, p = 3.5e-17; by-quintile p̂
rises monotonically hard→easy 0.074→0.104). "The teacher knows more than
it can act on" is confirmed — with the addendum that even what it knows
(live/dead) is only worth 1.5× under this utility at this pool.

Posterior fate 25→50 (both cells): only ~300 prompts visited in both
halves. Of live@25 revisited, p̂ drifted −0.02; 4–6% went dead. Of dead@25
revisited, 5–8% resurrected (p̂ > 0.1), mean α gain +0.34–0.43 — the model
is slowly igniting dead prompts, a real non-stationarity the teacher would
track if it ever got the visits.

---

## 2. Channel-1 headroom and fix ranking (scoped to a GSM8K re-run)

*Scope per coordinator: E-LLM-2 (tier posteriors + SFT warmstart) is
launched and frozen. This section is (a) narrative support for §7.5 and
(b) a GSM8K re-run design only.*

### The headroom measurements that reorder the ranking

Fitting a beta-binomial to the run's own 2001 1-visit outcomes (maxrl
cell; k̄ = 0.64/16, 67% dead draws) gives a true-p field Beta(0.34, 8.26).
Against that field:

| allocator | advantage mass vs uniform | dead-sampled |
|---|---|---|
| uniform | 1.00× | 0.686 |
| **per-prompt oracle** (exact p, +0.1 floor) | **1.88×** | 0.398 |
| oracle confined to the 7B-annotation axis (10 bins) | 1.04–1.05× | ~0.67 |
| E-LLM-1 as run (simulated) | 1.02× | 0.681 |
| E-LLM-1 as run (measured weights × axis utility) | 1.00–1.03× | 0.643–0.668 meas. |

The axis-aligned number is *empirical*, not just simulated: pooling both
cells' 1-visit observations (n=4018) by annotation decile, measured
u = pass@16 − pass@1 runs 0.217 (hardest decile) → 0.403 (easiest),
Spearman(decile, u) = 0.98 — beautifully monotone, but only a 1.85×
*spread*, which an optimal bin allocator converts into just **1.037×**
mass (1.033× with floor). And the annotations transfer weakly to 360M:
AUROC(7B solve% → observed 360M live-after-1-visit) = **0.593** (n=2560,
p=1.6e-15 — significant, feeble). Nearly all utility variance at 360M is
*within*-decile (idiosyncratic per-prompt), not along the 7B axis.

Simulated 50-step × 64 runs against the fitted field, annotation noise
calibrated to the measured AUROC (5 seeds):

| configuration | mass vs uniform | dead-sampled |
|---|---|---|
| as run (7473 pool, prompt-level, decay .7) | 1.02× | 0.681 |
| (iv) annotation prior s=16 | 1.04× | 0.672 |
| (i) 10 annotation bins | 1.03–1.05× | 0.669–0.677 |
| (ii) 1500-prompt pool, no decay | 1.12× | 0.645 |
| (iii) 150 steps, full pool | 1.07× | 0.663 |
| (ii)+(iii) 1500 pool, 150 steps | 1.29× | 0.588 |
| 1500 pool + **self-measured presweep prior** | 1.23× | 0.603 |
| 1500 pool + presweep + 150 steps | **1.37×** | 0.556 |
| per-prompt oracle bound | 1.88× | 0.398 |

("Presweep" = one 16-rollout generation-only sweep of the pool before
training, warm-starting α,β with the model's OWN pass counts — a perfect
1-observation prior. Cost at 1500 prompts ≈ 23 step-equivalents of
generation, ~+30% of a 50-step cell; at 7473 prompts it costs more than
the run.)

### Ranking (expected impact / A10G cost)

**Rank 1 — (ii) Subsample the pool to ~1–1.5k prompts, and slow the decay
(≥0.9 or off).** Zero extra GPU; the largest single-lever gain (1.02→1.12×
at 50 steps) and the enabler for everything else. Honest framing writes
itself: "prompt-level posteriors need visits/prompt ≥ 2; we set the pool
size so the budget provides it." Subsampling uniformly keeps both arms'
prompt diversity matched, so the comparison stays clean.

**Rank 2 — self-measured presweep prior (the correct version of fix iv).**
+~30% cell cost at a 1.5k pool for 1.12→1.23× (and 1.37× if combined with
150 steps). External annotations are NOT the way to warm-start here —
measured ceiling 1.04× (AUROC 0.59 transfer, compressed axis). The model's
own 16-shot pass rates are the informative prior that actually pays.
Bonus: the presweep gives exact day-0 dead/live labels, making the P-G4
steering meter trivially well-defined.

**Rank 3 — (iii) 150+ steps, ONLY on the subsampled pool.** At the full
pool it buys 1.07× for 3× the GPU (worst impact/cost of the structural
fixes); at 1.5k it compounds to 1.29–1.37×. Requires the decay change
(Rank 1) or the asymptotic 3×-odds cap of §1b binds.

**Rank 4 — (i) difficulty-bin posteriors / (iv) external-annotation
priors: DO NOT spend GSM8K GPU on these.** Measured axis ceiling 1.04×.
Important nuance for the paper: this does *not* indict E-LLM-2's tier
design — Countdown's tiers have a huge real pass-rate spread (probe
0.555/0.203/0.086), i.e. the between-tier utility variance IS the signal
there. GSM8K@360M's external axis carries almost none of the utility
variance. Tiers work when the axis does; the axis must be validated first
(the AUROC + decile-u table above is the 30-minute CPU check to run before
any binned design).

**Rank 5 — (v) kernel/shared posterior over a difficulty embedding
(streaming.py).** Statistically the right object only if a better axis
than the 7B annotations exists (embedding-based). As long as the axis is
this weak, it inherits the Rank-4 ceiling with extra machinery. Defer to
E-LLM-3.

**Rank 6 — (vi) floor/γ and (vii) larger teacher batch fraction:
refuted.** Floor contributes ~10% of weight (§1a); γ>1 amplifies SNR≈1
noise (§1d); giving a 0.977-ESS sampler more batch share scales a ~zero
signal.

**Also required in any re-run (validity, not performance):**
- A **uniform-with-replacement control cell** (see §3 — the sampler's
  with-replacement property is currently confounded with the teacher).
- Dump per-prompt success counts in the k-sweep (one-line change to
  `ksweep_eval.py`) to enable paired tests.
- The realized-mass meter: log Σu over sampled prompts per step vs a
  uniform-counterfactual — the 1.88× oracle bound makes this the honest
  primary meter for channel 1, with val as the downstream check.

### What this means for §7.5's narrative

Even a perfect per-prompt allocator gets 1.88× mass and dead-sampled 0.40
at this (model, pool). The maze gave the teacher ~40 visits/bin *and* a
utility spread of orders of magnitude across bins; GSM8K@360M gives 0.4
visits/prompt *and* a compressed, axis-orthogonal utility field. P-G1's
null needed no bad luck. Recommended one-sentence upgrade: "the teacher's
realized policy was a 1.2× visit tilt toward the third of the pool it had
seen, with sampling-weight ESS 0.98 — and the oracle ceiling itself was
only 1.9× at this model scale."

---

## 3. P-G2 confound audit (grpo+teacher regression 25→50)

Timeline (git + ray session starts): sampler per-chunk redraw fix c71cbe7
07-24 03:59; workers=0 launcher 830af9e 07-24 06:08. grpo_curtrue ran
fresh 1→50 starting 07-24 07:12; grpo_curfalse fresh 1→50 starting 07-25
01:19; both on the same commit state, identical launcher, differing only
in CURRICULUM. maxrl_curtrue's final run was also fully post-fix (07-25
19:24; the pre-fix partial was discarded). The only *resumed* cell is
maxrl_curfalse (25→50, WORKERS=2 per 7370b35) — touches P-G1's uniform
arm bookkeeping, not P-G2. **The mid-experiment sampler fix does not
confound P-G2.**

Three genuine caveats the analysis has not yet surfaced:

1. **The 25→50 regression is within eval noise.** Δ = −.003 mean@4 on a
   256-prompt × n=4 stochastic eval (sd of an unpaired difference ≈
   0.013). The sign claim "only cell that got worse" is a coin-flip
   statistic alone. What actually carries P-G2 is the endpoint gap
   (.120 vs .093 mean@4; .229 vs .181 pass@4 ≈ 2σ unpaired) plus the
   train-AUC gap (.0432 vs .0404). §7.5 should lean on those.
2. **With-replacement sampling is a bundled variable.** CurriculumSampler
   draws with replacement; uniform arms use verl's epoch-shuffled
   without-replacement sampler. Uniform arms saw 3200 distinct prompts;
   teacher arms ~2561 distinct with ~560 repeats. Prompt repetition under
   GRPO could hurt independently of *which* prompts are repeated. The maze
   H6 didn't carry this confound (both maze arms sampled with
   replacement). Cheap kill-shot: one 50-step GRPO cell with
   uniform-with-replacement. Until then, P-G2's mechanism sentence should
   say "frontier-weighted with-replacement sampling," not imply the
   difficulty-ordering alone.
3. **The dead-sampled steering evidence is overstated as quoted.** The
   "0.48 vs ~0.65 population" line compares the teacher cell's per-step
   *minimum* against a population *mean*. Honest same-statistic contrasts:
   step-means 0.643 (maxrl+cur) and 0.668 (grpo+cur) vs 0.667 (grpo
   uniform); minima 0.484/0.508 vs 0.516 (grpo uniform's own minimum).
   For the GRPO pair the means are indistinguishable (0.668 vs 0.667).
   The ~0.02 mean reduction in the maxrl cell is consistent with the 1.2×
   live tilt — the teacher steered *that* much and no more. Recommend
   P-G4's status line be weakened from "CONFIRMED post-fix (min 0.48 vs
   0.65)" to the mean-vs-mean statement; this also aligns GSM8K_ANALYSIS
   Finding 1's "verifiably live" phrasing and PAPER §7.5 with the
   artifact.

Also noted: step-0 val spread across cells (.066–.091, same warmstart) is
pure eval stochasticity — endpoint comparisons inherit ±.01 of start
noise; a second reason to move P-G2's weight onto paired per-prompt tests
(§4-A) and train-side meters.

### k-sweep (arrived 18:28 today) — P-G3 read

`ksweep_results.json` (vllm, n=16, temp 0.6, 256 rows): pass@16 = .238
(grpo), .215 (grpo+cur), .230 (maxrl), .215 (maxrl+cur). Two flags before
this goes near the paper:

- **Level discrepancy vs trainer val:** k-sweep pass@1 = .028–.038 vs the
  trainer's step-50 mean@4 = .093–.120 — a ~3× gap on nominally the same
  slice/temperature. Candidate causes: chat-template application
  difference, answer-extraction difference (ksweep regex vs reward
  manager), or the FSDP→HF bf16 export. Reconcile before quoting any
  k-sweep number (a per-prompt dump would diagnose this too).
- **n=256 noise:** se(pass@16) ≈ .026/cell, so the grpo-pair gap (.024)
  and maxrl-pair gap (.016) are each within ~1σ unpaired. As it stands,
  P-G3 ("divergence clearer than mean") is *not confirmed* — orderings
  match P-G2's direction but both teacher arms sit below both uniform
  arms at every k, which reads as much "with-replacement/teacher costs
  coverage under both estimators" as "GRPO-specific amplification."
  Paired per-prompt analysis (free once counts are dumped) or a larger
  eval slice is needed before P-G3 earns a scoreboard checkmark.

---

## 4. No-GPU measurements for §7.5 — status

- **(A) Paired per-prompt val / improvement-vs-visit correlation:**
  NOT POSSIBLE from existing artifacts — neither the trainer val loop nor
  ksweep_eval.py persisted per-prompt outcomes (ksweep computes them and
  writes only means), and the val slice is held-out so visit correlation
  isn't defined on it anyway. Action: one-line ksweep change (dump per-
  prompt c) + rerun in the next GPU window; then McNemar on the 256
  shared prompts turns the .120-vs-.093 endpoint into a paired claim.
- **(B) Annotation-prior go/no-go: DONE — verdict NO-GO** for external
  annotations as priors (AUROC 0.593; day-1 dead-sampled improves 0.626 →
  0.622 simulated; mass ceiling 1.04×). The decile calibration table
  (implied 360M p: 0.019 hardest-decile → 0.048 easiest) is itself a
  §7.5-worthy exhibit: the 360M frontier is nearly orthogonal to the 7B
  difficulty axis. Self-measured presweep replaces it (§2 Rank 2).
- **(C) Realized channel-1 ceiling: DONE** — as-run mass 1.00–1.03× vs
  1.88× per-prompt oracle vs 1.04× axis oracle. The single most
  paper-strengthening number set: it converts "the teacher didn't help"
  into "the teacher captured ~3% of a ~90% possible mass gain, of which
  ~0% was reachable along any external difficulty axis."
- **(D) First/second-half allocation: DONE** — Spearman ≈ 0 both halves
  on the axis, but class-level tilt 1.19–1.24× toward live@25 in the
  second half (§1e). This is the direct, artifact-level proof of
  "learns but cannot act," stronger than the flat-quintile figure alone.
- **(E) Dead-sampled trajectory: DONE** — quote means (0.643/0.668 vs
  0.667), not minima-vs-population (§3.3).

---

## A. Methods appendix

- Teacher checkpoints:
  `~/ckpt/gsm8k_a10g/{maxrl,grpo}_curtrue/global_step_{25,50}/curriculum_teacher.pt`
  (dicts of alpha/beta/visits; torch.load weights_only=False).
- Annotations: lime-nlp/GSM8K_Difficulty (HF cache), 7470/7473 matched by
  normalized-prefix of the user-turn question. Difficulty := −solve%.
- Utility MC: M=400 Thompson draws p̃ ~ Beta(α,β);
  u = max(pass@16(p̃) − p̃, 0); weights = 0.9·u/Σu + 0.1/n, exactly
  matching `verl/utils/curriculum.py::FrontierTeacher.sampling_weights`.
  ESS = 1/(n·Σw̄²).
- True-p field: beta-binomial moment fit to the 2001 one-visit success
  counts (k̄=0.637, intra-group correlation ρ=0.104) → Beta(0.343, 8.260). Empirical
  decile-u uses pooled 1-visit data from both cells (n=4018);
  u_decile = mean(k>0) − mean(k)/16.
- Fix simulations: exact replica of the FrontierTeacher update loop
  (per-observation decay, floor 0.1, Thompson, with-replacement chunk
  draws), 50 or 150 steps × 64 groups, 3–5 seeds; annotation proxy =
  log p + N(0, 8²), calibrated so AUROC(proxy → live-after-1-visit) =
  0.59, matching the measured value.
- Trajectories: `maxrl/curriculum_maxrl/gsm8k_partial/cell_trajectories.json`
  (dead-sampled = verl's `fraction_of_prompts_in_[0.0, 0.0]`);
  k-sweep: `.../ksweep_results.json` (written 2026-07-27 18:28).
- Timeline: git log of `maxrl` (c71cbe7 → 830af9e → bc4d05f → 7370b35)
  cross-checked against ray session start times in cell_trajectories.
