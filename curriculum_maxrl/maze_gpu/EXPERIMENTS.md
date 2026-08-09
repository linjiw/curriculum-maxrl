# maze_gpu experiment log

> **Historical-audit warning (2026-07-21).** All tables below predate several
> corrections: the old zero-weight counter mixed all-fail K=0 and all-pass
> K=N groups; the frontier teacher used legacy `u_{N+1}`, not exact `u_N`;
> hindsight called legal path length “depth” even when paths looped;
> all training levels used the deepest response budget although evaluation
> used level-specific budgets; dense hindsight grew in magnitude with relabel
> count; and dead-group values were sparse evaluation-step snapshots, not
> run-wide rates. Evaluation and teacher logging also consumed the subsequent
> training/curriculum RNG streams; shared warmstarts matched weights but not
> post-SFT random streams; and the time-limit record used an incremented,
> untrained step index. The code now fixes these contracts. Treat the numbers below
> as exploratory provenance pending a corrected rerun. Historical AUCs were
> unanchored and integrated over optimization step despite wall-clock-matched
> endpoints; `analyze.py` now anchors the post-SFT point and reports total
> process wall-clock and step AUC separately.

GPU testbed: 1.26M-param decoder-only transformer (6 layers, d=128) on 17×17
Prim mazes, A10G. Difficulty = BFS distance of the goal from start (1,1):
13 levels, distance 4,6,…,28. Binary verifier on the emitted move string.
Infinite-data regime (fresh mazes每 step), matching the paper's maze setup.

## Archived pilot observations (raw drivers/artifacts not retained)

1. **Maze *size* is a broken curriculum dimension.** After SFT on 5×5/7×7,
   9×9 pass rate is exactly 0/1024 — a hard generalization cliff (different
   prompt geometry). Goal-*distance* within a fixed 17×17 grid transfers:
   post-SFT pass decays smoothly (0.98, 0.92, 0.34, 0.16, 0.10, 0.03, →0).
2. **SFT mixture shape matters.** SFT on levels {0,1} only → cliff at level 2.
   Geometric mixture (weight 0.5^level) gives the smooth frontier above while
   still leaving levels ≥6 at p≈0 (the curriculum question stays real).
3. **One training process per A10G.** Two concurrent runs OOM at generation
   time (batch 256 × 308-token prompts).
4. Batched generation across all groups per step: ~2.5 s/step at
   8 tasks × 32 rollouts; micro-batched backward (128 rows) bounds memory.

## Sweep 1 (seed 0, 300 steps, 8 tasks/step × 32 rollouts, lr 1e-4)

Configs: uniform/frontier/learnability × maxrl; uniform/frontier × grpo;
frontier × rloo; frontier_alp × maxrl. Results: see `analyze.py` output
appended below when complete.

Post-SFT baseline eval (level: pass@sampled):
0:0.95 1:0.74 2:0.48 3:0.27 4:0.07 5:0.04 6–12:0.00

### Sweep 1 results (300 fixed steps — superseded, see protocol note)

| config | dead/8 | wall-clock for 300 steps | final mean | best mean |
|---|---|---|---|---|
| uniform+maxrl | 5.2 | 2970 s | 0.224 | 0.231 |
| frontier+maxrl | 3.9 | **1477 s** | 0.206 | 0.242 |
| learnability+maxrl | 2.6 | **1245 s** | 0.225 | 0.237 |

Three mechanism findings:

1. **The old zero-weight snapshots cannot identify dead-group waste.** They
   mixed K=0 and K=N, so 65%/49%/33% are provenance, not corrected dead rates.
2. **Teachers were ~2× faster per step in the old stack.** Deep-level rollouts
   used the global deepest response budget (37 tokens); frontier/learnability sample frontier levels
   whose successful paths are short, halving generation time. **Fixed-step
   comparison is therefore unfair to teachers** — switched to matched
   post-SFT process time including evaluation (2400 s per config,
   `--max-seconds`).
3. **Teacher posteriors were qualitatively ordered, not tightly calibrated.**
   The historical p_hat roughly preserved the frontier ordering and put 60%
   of mass on levels 2–5, but final per-level errors reach roughly 0.1; there
   is no supported ±0.03 accuracy bound.

At equal *steps*, all three tie within noise (~0.22). The historical
matched-clock sweep (`matched_*.jsonl`) compares 6 configs =
{uniform, frontier, learnability, frontier_alp} × maxrl + {uniform, frontier} × grpo.

Protocol note: pass@k eval (unbiased Chen et al. 2021 estimator, k∈{1,8})
was added to `evaluate()` while config 1 (uniform+maxrl) was already running,
so its log lacks `passk` records; configs 2–6 have them.

## Historical matched-time results (2400 s post-SFT process time including evaluation, seed 0)

| config | steps | old zero-weight snapshot/8 | final | best | legacy unanchored step-AUC | pass@8 final | frontier |
|---|---|---|---|---|---|---|---|
| uniform+maxrl | 583 | 5.8 | 0.225 | 0.233 | 0.214 | — | 2 |
| uniform+grpo | 527 | 6.0 | 0.230 | 0.237 | 0.216 | 0.312 | 2 |
| frontier+grpo | 751 | 5.7 | 0.224 | 0.249 | 0.219 | **0.269** ↓ | 3 |
| uniform+maxrl+hindsight | 651 | 5.0 | 0.233 | 0.237 | 0.222 | 0.356 | 2 |
| frontier+maxrl | 713 | 3.9 | 0.237 | 0.240 | 0.223 | 0.351 | 2 |
| learnability+maxrl | 787 | 3.3 | 0.227 | 0.250 | 0.227 | 0.356 | 2 |
| frontier_alp+maxrl | 765 | 3.4 | **0.244** | **0.257** | 0.233 | **0.361** | 2 |
| frontier+maxrl+hindsight | 694 | 3.8 | 0.230 | 0.256 | **0.234** | 0.356 | 3 |

(single seed — treat orderings within ±0.01 AUC as ties)

**Findings:**

1. **Every teacher/hindsight variant has higher legacy step-AUC than both uniform baselines in this seed.**
   Top two: frontier+maxrl+hindsight (0.234) and frontier_alp+maxrl (0.233)
   vs 0.214/0.216 uniform. The old stack shows more steps per second and fewer
   zero-weight snapshots, but the audit prevents a mechanism attribution.
2. **H6 historical direction reversal.** Prediction was that the teacher
   patches GRPO's pass@k collapse by retiring mastered prompts. Instead
   frontier+grpo collapsed *more* (pass@8 0.332→0.269) than uniform+grpo
   (0.351→0.312), and lost easy-level retention (min easy pass 0.62 vs 0.75).
   One possible reading is that easy-prompt updates helped maintain coverage,
   but the audited stack does not isolate that mechanism. This is a hypothesis
   about objective–curriculum interaction, not a general conclusion. MaxRL configs under the
   same historical teacher keep pass@8 flat-to-up (0.327→0.351).
3. **The historical GPU hindsight delta is much smaller than CPU** (legacy AUC +0.008/+0.011
   vs +0.22 on the toy): relabeled mazes here are one-off (fresh maze每 step,
   no repeated task to cash in the relabeled skill), and the relabeling only
   fires on the single best rollout per dead group. Both hindsight configs
   tie for best frontier depth (level 3) and best pass@8 at their band.
4. **frontier_alp has the highest seed-0 historical pure-teacher point
   estimate** (level-2 pass 0.62 vs 0.54 frontier, best final overall). The
   audited single-seed comparison does not establish an ALP effect.

## Historical multi-seed check (seeds 0–2, matched 2400 s, key configs)

| config | legacy unanchored step-AUC (3 seeds) | final | pass@8 first→last | ΔAUC vs uniform+maxrl per seed |
|---|---|---|---|---|
| uniform+maxrl | 0.211 ± 0.011 | 0.230 ± 0.015 | 0.310 → 0.300 | — |
| uniform+grpo | 0.213 ± 0.006 | 0.230 ± 0.001 | 0.308 → **0.271** ↓ | mixed (+0.003, −0.005, +0.008) |
| frontier_alp+maxrl | 0.221 ± 0.013 | **0.246 ± 0.002** | 0.306 → 0.338 ↑ | **all positive** (+0.019, +0.004, +0.006) |
| frontier+maxrl+hindsight | **0.223 ± 0.010** | 0.234 ± 0.005 | 0.316 → **0.348** ↑ | **all positive** (+0.020, +0.002, +0.013) |

Seeds share their SFT warmstart, so per-seed deltas are paired comparisons.

**Observed across three seeds (not confirmatory):**

1. Both teacher variants have positive legacy AUC deltas in every seed. The
   six deltas are two correlated configurations over only three independent
   seeds; frontier_alp also has higher final mean-pass with the
   tightest spread (0.246 ± 0.002 vs 0.230 ± 0.015).
2. The observed pass@k divergence is a rerun hypothesis: GRPO *decays*
   coverage in every seed (mean 0.308 → 0.271) while teacher+MaxRL configs
   *grow* it (0.306→0.338, 0.316→0.348). Three seeds and the audit confounds
   do not separate this pattern from seed noise or implementation effects.
3. uniform+grpo ≈ uniform+maxrl on mean pass at this scale — the estimators
   separate on *coverage* (pass@8) and on how they respond to a curriculum,
   not on average performance.

## FrontierMax v2: closing the frontier gap (design + rationale)

**Diagnosis of the wall.** Best config (frontier_alp) ends at
p̂ ≈ [0.94, 0.87, 0.55, 0.35, 0.27, 0.07, ~0…]: level 6+ never leaves 0.
At the observed level-6 pass rate (~0.005), even pass@128 ≈ 0.47 — brute
rollouts can't cross. But the raw material is there: 33% of failed level-6
rollouts legally reach depth ≥ 10, and the current sparse hindsight harvests
only **1 relabel per dead group (3.6/step)** out of up to 32.

**v2 changes (all in `train.py`):**

1. `--hindsight-dense`: relabel *every* failed rollout with legal prefix
   ≥ `--hindsight-min-depth` (default 6) to the cell it reached, capped at
   `--hindsight-cap` (16) per step → ~10× more salvaged signal per dead
   group, each a verifier-valid (prompt, trajectory) pair; this does not imply
   equality to a fresh-task gradient law.
2. `--hindsight-to-teacher`: relabeled successes update the teacher
   posterior at the matching distance level. Rationale: hindsight teaches
   deep-navigation skill that the teacher's posterior never sees (it only
   observes original-task rewards), so the curriculum lags the student's
   true frontier. Deliberately optimistic (reached *some* cell at distance
   d, not a requested one); posterior decay corrects overshoot.
3. Archived CPU A/B on goal selection (driver/output not retained): relabeling to the *deepest* reached prefix
   beats picking the advantage-mass-optimal prefix (AUC 0.863 vs 0.848) —
   when signal is free, take the most of it; mass-optimality matters only
   when allocation is the scarce resource. GPU version keeps deepest-cell.

**A/B/C protocol (matched 2400 s, frontier_alp teacher):**
A = sparse hindsight (current), B = dense, C = dense + teacher feedback.
Success criterion: level-6 pass rate leaves 0 and/or AUC > 0.234 (current
best); watch for the failure mode where optimistic posterior updates drag
sampling beyond the true frontier (dead-group rate would rise).

**A/B/C RESULTS (complete):**

| config | final | best | legacy unanchored step-AUC | pass@8 | frontier | relabeled/step |
|---|---|---|---|---|---|---|
| baseline (frontier_alp, no hs) | 0.244 | 0.257 | 0.233 | 0.361 | 2 | 0 |
| A sparse hindsight | 0.226 | 0.254 | 0.234 | 0.365 | 2 | 3.6 |
| **B dense hindsight** | **0.258** | **0.269** | 0.236 | 0.361 | 2 | **16.0** |
| C dense + teacher feedback | 0.242 | 0.260 | **0.237** | 0.361 | **3** | 15.9 |

1. **Dense hindsight (B) has the best historical point estimates**: final (0.258) and
   peak (0.269) of every GPU config to date; level-0–3 pass rates all
   improve simultaneously (0.99/0.87/0.68/0.45) — the relabeled gradients
   strengthen shallow navigation without sacrificing the frontier.
   Harvest rate went 3.6 → 16.0 relabels/step (the cap), exactly the
   ~4.4× the design predicted. Its loss also scaled with that count, so this
   comparison confounds relabel coverage and gradient magnitude.
2. **Teacher feedback (C) directionally matched the V4 prediction**: historical AUC ties B
   (0.237 vs 0.236, noise), final is lower (0.242), and the mechanism's
   signature is visible — C's posterior at level 2 inflates to p̂=0.81 vs
   eval 0.47 (B tracks: 0.54 vs 0.64), pushing sampling mass deeper
   (earning the only frontier=3 flag) at the cost of consolidating level 2.
   The old counter cannot establish “no runaway,” and protocol confounds
   prevent a policy verdict. For the corrected rerun, requested-task-only
   feedback remains the conservative default and teacher feedback is an
   explicit ablation.
3. Level 6 still ≈ 0.01–0.02: dense hindsight lifts the *approach* to the
   frontier but one 2400 s budget doesn't cash it out at distance 16+.
   The efficiency study + longer runs are the follow-up.

## Archived inference-efficiency study (E4; not reproducible as shipped)

Three historical matched-time checkpoints were evaluated with unbiased pass@k
from 64 samples × 16 mazes/level. The checkpoint files were not retained, and
the table used different post-hoc coverage targets by level even though the
current evaluator takes one global target. Treat these values as descriptive
archive only, not a reproducible or preregistered comparison:

| level (target) | GRPO k* | MaxRL-uniform k* | ours (teacher+dense-hs) k* | **ours vs GRPO** |
|---|---|---|---|---|
| 2 (85%) | 7.5 | 7.2 | 6.4 | 1.2× |
| 3 (75%) | 39.3 | 13.5 | 14.8 | **2.7×** |
| 4 (45%) | 6.7 | 10.7 | 12.8 | 0.5× |
| 5 (25%) | >64 → 64.0 | 10.5 | **5.8** | **11×** |

The historical point estimates range from a 0.5× reversal at level 4 to an
11× advantage at level 5, so they do **not** establish a monotone
difficulty–speedup relationship. Apparent curve flattening and tail coverage
are hypotheses for a corrected, seeded, common-target evaluation with retained
checkpoints; no multiplier from this table is current evidence.

## F3/F4 historical multi-seed point estimates (seeds 0–2, matched 2400 s)

| config | final (3 seeds) | legacy unanchored step-AUC (3 seeds) | pass@8 |
|---|---|---|---|
| uniform+maxrl | 0.230 ± 0.015 | 0.211 ± 0.011 | 0.300 |
| frontier_alp+maxrl | 0.246 ± 0.002 | 0.221 ± 0.013 | 0.338 |
| **falp + dense hindsight** | **0.252 ± 0.005** | **0.229 ± 0.009** | 0.335 |

Paired per-seed deltas (shared warmstarts), dense hindsight − frontier_alp:
Δfinal = +0.014/+0.002/+0.001, ΔAUC = +0.003/+0.006/+0.014. These are two
metrics over only three paired seeds, without a formal test, and dense loss
scaled with relabel count. The point estimates favor dense hindsight, while
the final margin is mostly one seed; reliability remains unestablished.
(consistent with the one-shot-mazes analysis — salvaged skill can't
compound on a task you never see again). Coverage is similar between the two teacher variants
(0.335 vs 0.338). The fixed-prompt-set regime (GSM8K) remains where dense
hindsight should show CPU-like compounding.

## Depth-mechanism study: the stall at distance 16, diagnosed

Probes on the long-horizon checkpoint answer F1's open question:

- **H-a (move budget binds): REFUTED.** Doubling the budget adds zero
  successes (L6: 0/128 both ways); 0/128 rollouts die by running out of
  budget — 124/128 die on an *illegal move*, 4 on premature EOS.
- **H-c (representation reach): CONFIRMED, quantitatively.** Per-step move
  legality is ≈0.87 after the opening (1.00 at move 0, 0.43–0.84 in moves
  2–10). Reach is therefore geometric: mean legal prefix 6.4–7.7, p90 ≈
  8–14, max ever observed 12–14 — matching E[reach] ≈ q/(1−q) ≈ 6.7 at
  q=0.87. **P(reach 16) ≈ q¹⁶ ≈ 1–3%**: the frontier march stalls exactly
  where compounding per-step error says it must.
- **Contributing factor:** SFT exposure was geometric (decay 0.5), so only
  1.6% of warmstart examples had length ≥ 16; RL+hindsight then mostly
  reinforces shallow segments (relabels land at depth 6–10, the reach
  distribution's mass).

**Conclusion: the stall is a per-step-accuracy (capacity/supervision)
ceiling, not a curriculum failure.** The curriculum correctly walked the
frontier to the edge of what the policy can execute; going deeper requires
raising per-step legality (q=0.87 → ≥0.95 for distance 16). This is the
compounding-error regime imitation learning knows well — and it sharpens
the paper story: a teacher can place signal optimally, but *cannot create
per-step competence beyond the model's ceiling*. Candidate fixes, in test
order: (1) capacity probe (wider model, same schedule — decisive single
run), (2) denser hindsight supervision (train all prefixes of a relabeled
trajectory, not just the endpoint).

**Capacity probe verdict (256×8 = ~5M params, champion schedule, matched
2400 s → only 339 steps at ~2× cost/step):**

| | 128×6 champion | 256×8 wide |
|---|---|---|
| AUC | 0.236 | **0.248** (record) |
| best mean | 0.269 | **0.274** (record) |
| level 3 final | 0.38–0.45 | **0.55** |
| level 6 final | 0.01–0.02 | **0.05** |
| per-step legality (L6) | 0.870 | 0.877 |

Partial confirmation with a twist. Capacity sets new AUC/best records
*despite 40% fewer steps* — per-step productivity way up, and levels 3–6
all lift (L6 leaves the 0.01 floor for the first time, ×2.5–5). But
per-step legality barely moved (0.870→0.877): the wide model is not yet
*executing* more accurately at depth; it is converting the same schedule
into better shallow/mid competence faster. Reading: capacity relieves the
ceiling gradually, not as a step change — level 6 mastery likely needs
capacity × longer training × deeper SFT exposure together. For the paper:
"curriculum + capacity" compose (best-ever numbers), and the diagnosis
methodology (measure q, predict reach geometrically) is itself a
contribution — it predicted both the stall and where capacity's gain would
land (throughput, not depth-execution).

## P1: efficiency of the long-horizon and wide checkpoints

Samples-to-target-coverage (same protocol as E4), plus the deep-frontier
coverage the training metrics can't see:

| level (target) | GRPO 2400s | champion 2400s | long 9600s | wide 2400s |
|---|---|---|---|---|
| 2 (85%) | 7.5 | 6.4 | >64 † | **1.3** |
| 3 (75%) | 39.3 | 14.8 | **4.0** | 28.9 |
| 4 (45%) | 6.7 | 12.8 | **3.2** | 7.4 |
| 5 (25%) | 64.0 | 5.8 | 6.1 | 64.0 |
| **L6 coverage @ k=64** | 0.125 | 0.188 | 0.312 | **0.438** |

† the long run's L2 regressed below 85% even at k=64 — the frontier-following
teacher had moved on from shallow levels by step 2381 and the floor alone
didn't fully maintain them (an ALP-retention data point at long horizons).

Two findings:

1. **The deep frontier is moving after all — in coverage currency.** Training
   pass@1 said L6 ≈ 0.01–0.05 everywhere; coverage@64 tells a different story:
   GRPO 0.125 → champion 0.188 → long 0.312 → **wide 0.438 (3.5× GRPO)**.
   The wide model at matched 2400 s puts nearly half of L6 within reach of
   64-sample inference with a verifier. The frontier march did not stall —
   it moved into the tail of the distribution where pass@1 can't see it.
   This is precisely the paper's diversity/coverage thesis at work in our
   own stack.
2. **Training regime specializes the inference profile.** The long run
   dominates mid-levels (L3/L4 at 4.0/3.2 samples — 10×/2× vs GRPO) but paid
   for it at L2; the wide model dominates shallow+deep (L2 at 1.3, L6 at
   0.438) with a mid-level dip. Budget shape (duration vs capacity) is itself
   a curriculum-outcome knob. For deployment: pick the checkpoint by the
   difficulty band you'll serve — or ensemble the two.

## F1/F2 verdicts (final sweep)

**F1 — level 6 is NOT (just) a duration question.** 4× budget (9600 s, 2381
steps): mean climbs 0.258→0.269 and level 5 doubles (0.17→0.23–0.25), but
level 6 stays ≈0.01–0.02 the entire run. The frontier march decelerates
hard between distance 14 and 16. Per the pre-registered decision tree, the
mechanism needs revision at depth: candidates are (a) move budgets that
scale with achieved depth (the dist+8 cap may bind exploration wander),
(b) hindsight-min-depth curriculum, (c) the MountainCar transfer lesson —
check whether tile/prompt representations even share parameters across
these depths. CPU-validate before spending GPU.

**F2 — γ=4 did not improve this historical maze run.** Legacy step-AUC 0.231 /
best 0.254 versus γ=1's 0.236/0.269. Weak compounding across 13 broad levels
is one hypothesis, but the audited GPU confounds prevent a mechanism claim.
Decision for the corrected maze rerun: retain γ=1 until a clean concentration
ablation says otherwise.

### Hypotheses for the matched-clock analysis

- **H6 (GRPO inversion fix).** The paper (Section 5, footnote 3) shows GRPO's
  w(p) *inverts* as p→1 — upweighting mastered prompts — and conjectures this
  drives distribution sharpening / pass@k collapse. A frontier teacher
  retires p≈1 prompts from the batch entirely, removing the regime where the
  inversion applies. Prediction: `frontier+grpo` shows better pass@8 (less
  collapse) than `uniform+grpo`, i.e. **a curriculum can patch GRPO's
  pathology at the data level without touching the estimator**.
- **H7 (matched-clock ordering).** frontier+maxrl > uniform+maxrl on mean
  eval and frontier depth, driven by ~2× more steps and ~35% fewer dead
  groups in the same wall-clock.

### Theory update (see ../THEORY.md)

The teacher utility is now derived from expected scalar coefficient mass:
`EΣ|w|=2(pass@N − pass@1)`, whose peak is p* ≈ ln(N)/N.
RLOO's expected mass is `2p(1−p)`, proportional to SFL learnability, and GRPO's realized
finite-sample mass on hard prompts is ~2× below its population w(p) due to
degenerate groups. With fixed known p, feasible integer bounds, and a one-step
budget, greedy water-filling on half-mass marginal `p(1−p)^N` is exact for
that proxy. CPU validation: advmass teacher ties frontier teacher
(AUC 0.704 vs 0.712, both > zpd 0.688), greedy allocation ≈ adaptive; the
derived form wins on principle (parameter-free) not performance.
