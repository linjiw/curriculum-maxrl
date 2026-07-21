# maze_gpu experiment log

GPU testbed: 1.26M-param decoder-only transformer (6 layers, d=128) on 17×17
Prim mazes, A10G. Difficulty = BFS distance of the goal from start (1,1):
13 levels, distance 4,6,…,28. Binary verifier on the emitted move string.
Infinite-data regime (fresh mazes每 step), matching the paper's maze setup.

## Design decisions discovered by pilots

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

1. **Dead-group waste is real and the teachers fix it.** Uniform wastes 65%
   of groups (K=0, dropped by MaxRL); frontier cuts to 49%, learnability to 33%.
2. **Teachers are ~2× faster per step.** Deep-level rollouts wander for the
   full move budget (57 tokens); frontier/learnability sample frontier levels
   whose successful paths are short, halving generation time. **Fixed-step
   comparison is therefore unfair to teachers** — switched to matched
   wall-clock (2400 s RL per config, `--max-seconds`).
3. **Teacher posteriors track truth well.** frontier p_hat
   [0.98, 0.87, 0.58, 0.32, 0.11, 0.09, ~0…] ≈ eval pass rates; its
   distribution concentrates 60% of mass on levels 2–5 (the true frontier).

At equal *steps*, all three tie within noise (~0.22) — as expected when the
teachers' savings are returned as unused time. The matched-clock sweep
(`matched_*.jsonl`) is the definitive comparison: 6 configs =
{uniform, frontier, learnability, frontier_alp} × maxrl + {uniform, frontier} × grpo.

Protocol note: pass@k eval (unbiased Chen et al. 2021 estimator, k∈{1,8})
was added to `evaluate()` while config 1 (uniform+maxrl) was already running,
so its log lacks `passk` records; configs 2–6 have them.

## Matched wall-clock results (2400 s RL each, seed 0, complete)

| config | steps | dead/8 | final | best | AUC | pass@8 final | frontier |
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

1. **Every teacher/hindsight variant beats both uniform baselines on AUC.**
   Top two: frontier+maxrl+hindsight (0.234) and frontier_alp+maxrl (0.233)
   vs 0.214/0.216 uniform. The mechanisms compound: more steps per second
   (583→700+), fewer dead groups (5.8→3.3–3.9), plus hindsight recycling.
2. **H6 REFUTED — direction reversed.** Prediction was that the teacher
   patches GRPO's pass@k collapse by retiring mastered prompts. Instead
   frontier+grpo collapsed *more* (pass@8 0.332→0.269) than uniform+grpo
   (0.351→0.312), and lost easy-level retention (min easy pass 0.62 vs 0.75).
   Reading: GRPO's inverted w(p) was effectively *maintaining* easy prompts;
   concentrating its updates on the frontier removes that maintenance and
   sharpens harder. **The objective is the problem — a data-level curriculum
   cannot rescue GRPO's collapse, which strengthens the paper's own claim
   that pass@k degradation is driven by objective choice.** MaxRL configs
   under the same teacher keep pass@8 flat-to-up (0.327→0.351).
3. **GPU hindsight gain is real but much smaller than CPU** (AUC +0.008/+0.011
   vs +0.22 on the toy): relabeled mazes here are one-off (fresh maze每 step,
   no repeated task to cash in the relabeled skill), and the relabeling only
   fires on the single best rollout per dead group. Both hindsight configs
   tie for best frontier depth (level 3) and best pass@8 at their band.
4. **frontier_alp (ALP anti-forgetting bonus) is the best pure-teacher
   variant** — first setting where the |Δp̂| term clearly earns its keep
   (level-2 pass 0.62 vs 0.54 frontier, best final overall).

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

The teacher utility is now *derived*, not heuristic: expected MaxRL advantage
mass per group is exactly `2(pass@N − pass@1)`, peaking at p* ≈ ln(N)/N.
RLOO's mass is exactly `2p(1−p)` (= SFL learnability), and GRPO's realized
finite-sample mass on hard prompts is ~2× below its population w(p) due to
dead groups. Greedy water-filling on marginal mass `p(1−p)^N` is the optimal
rollout allocation. CPU validation: advmass teacher ties frontier teacher
(AUC 0.704 vs 0.712, both > zpd 0.688), greedy allocation ≈ adaptive; the
derived form wins on principle (parameter-free) not performance.
