# Design Guide: approaches, verification status, and roadmap

This is the working map of the project: every approach we proposed, how far each
has been verified, and what's queued next. Detailed derivations live in
`curriculum_maxrl/THEORY.md`; per-experiment tables in `curriculum_maxrl/DESIGN.md`
and `curriculum_maxrl/maze_gpu/EXPERIMENTS.md`.

## 1. Problem framing

MaxRL Eq. (9) averages successful score functions and is unbiased for the
T=N-truncated maximum-likelihood objective. Full Eq. (10) preserves that
expectation with an unconditional score control. The practical Algorithm 1,
which this repository implements, drops the entire K=0 group; its exact
population objective is T=N-1. It remains an *implicit curriculum at the
gradient level*. Three gaps remain that only a *data-level* teacher can close:

1. **Dead prompts** — p ≪ 1/N ⇒ all N rollouts fail ⇒ K=0 ⇒ group dropped, zero
   gradient. No choice of w(p) can put signal where groups die.
2. **Mastered prompts** — p→1 ⇒ ∇p→0; rollouts are wasted compute (GRPO is worse:
   its w(p) *inverts* and upweights p→1, the paper's conjectured cause of pass@k
   collapse).
3. **Uniform rollout budgets** — the paper uses fixed N per prompt, but harder
   prompts need larger N both for signal probability and practical objective
   fidelity (T=N-1; exact Eq. 10 would give T=N).

Our thesis: **the right curriculum signal falls out of the estimator's own algebra**
rather than needing an external heuristic.

## 2. Proposed methods and verification status

### M1. Advantage-mass teacher (core method) — ✅ derived + validated (CPU and GPU)

- **Claim (proved, MC-verified 200k trials):** expected coefficient L1 mass a
  prompt receives from a MaxRL group of N rollouts is exactly
  `2(pass@N − pass@1)`;
  peaks at p* ≈ ln(N)/N.
- **Method:** decayed Beta posterior over each prompt's pass rate, updated from
  observed group rewards; Thompson-sample p; sample prompts ∝ `(1−(1−p)^N) − p`;
  uniform floor (default 0.1) for coverage/anti-forgetting.
- **Verification:**
  - Formula: Monte-Carlo match to 3 decimals across p ∈ [0.005, 0.95] (THEORY.md §2, §5).
  - CPU skill-chain (5 seeds): AUC 0.704 vs 0.712 heuristic-frontier vs 0.688
    hand-tuned ZPD — ties the best heuristic with zero band hyperparameters.
  - Posterior fidelity (GPU maze): teacher p̂ tracks true eval pass rates to ±0.03;
    concentrates 60% of sampling mass on the true frontier band.
  - Dead-group reduction (GPU maze): 5.2/8 dead groups per step under uniform →
    3.9 (frontier) / 2.6 (learnability-style) under teachers.
- **GPU verdict:** the matched-wall-clock sweep and three-seed confirmation are
  complete. Both teacher configurations beat uniform on final score and AUC;
  dense hindsight adds a reliable AUC gain over the plain teacher.

### M2. Greedy rollout allocation — ✅ derived + validated (CPU-level)

- **Claim (proved):** maximizing total advantage mass Σᵢ[(1−(1−pᵢ)^{Nᵢ}) − pᵢ]
  s.t. ΣNᵢ = B is concave per prompt ⇒ greedy water-filling optimal. Marginal
  value of the (N+1)-th rollout on prompt i is `pᵢ(1−pᵢ)^N` — the probability
  that rollout is the group's **first success**.
- **Verification:** +18% total advantage mass vs uniform split on a
  mixed-difficulty batch; small consistent AUC gain over the older 1/p̂ heuristic
  on the skill-chain testbed.
- **Open:** requires per-prompt `rollout.n` support in the verl rollout worker
  (phase-2 integration); untested at LLM scale.

### M3. Exact learnability equivalences (analysis) — ✅ proved

- RLOO's advantage mass is exactly `2p(1−p)` — identical (up to constant) to SFL
  "learnability" (Rutherford et al. 2024). The learnability-curriculum literature
  and the RLOO estimator are the same object from two sides.
- GRPO's *realized* finite-sample mass on hard prompts is ~2× below its population
  w(p) (at p=0.01, N=32: 0.100 vs 0.199) because 72% of groups are all-fail.
  The paper's population-level weight curves describe the infinite-sample limit only.
- Connection to SEC (Chen et al. 2025c): SEC drives a curriculum bandit with
  empirical |advantage| as reward; our formulas are the exact expectations of
  SEC's signal — the advantage-mass teacher is "oracle SEC for MaxRL," usable
  before a prompt is ever visited.

### M4. Adaptive truncation order T (objective curriculum) — ⚠️ exact path untested

- **Audit result:** `c_sub_TN` makes the success term unbiased for arbitrary
  T<=N, but the historical `weights_maxrl_t` then dropped the K=0 score
  control. Its exact multiplier is `w_T(p)-(1-p)^(N-1)`, not `w_T(p)`.
- **Historical result:** that dropped-group variant slightly underperforms its
  fixed counterpart (AUC 0.698 vs 0.704 with advmass; 0.641 vs 0.653 uniform).
  This is a valid negative for the tested heuristic, not for exact adaptive T.
- **Status:** `weights_maxrl_t_eq10` now retains the control term for K=0 and
  passes exhaustive finite-N tests. It has no learning experiment yet.

### M5. ALP anti-forgetting term — ⚠️ implemented, weak evidence

- `FrontierALPTeacher` adds an ALP-GMM-style |Δ ema-pass| bonus to re-inject
  regressing levels. Pure-ALP teachers underperformed on the CPU testbed, but
  the additive GPU variant is the best pure teacher: final 0.246 ± 0.002 and
  AUC 0.221 ± 0.013 over three seeds. Long-horizon easy-level regression shows
  that the floor alone does not guarantee retention indefinitely.

### M7. Hindsight relabeling for dead groups — ✅ validated (CPU and GPU)

- **Idea:** MaxRL's target is success-conditioned. Its variance-reduced weights
  include failures as a control variate inside live groups, but K=0 groups have
  exactly zero update. HER's move is the complement: a failed trajectory can be
  a success **for the goal it actually reached**. Where task structure admits
  relabeling, every dead group converts into a live group for an easier related
  task at zero extra generation cost.
- **CPU result (5 seeds, skill-chain; failed level-l rollout with correct prefix
  j = success of nested level-j task):**

  | config | final | AUC |
  |---|---|---|
  | uniform+maxrl | 0.966 | 0.653 |
  | uniform+maxrl+**hindsight** | 0.978 | **0.878** |
  | advmass+maxrl | 0.979 | 0.704 |
  | advmass+maxrl+**hindsight** | 0.984 | **0.883** |

  **Largest single improvement in the project** — bigger than the teacher itself
  on learning speed (AUC +0.22 vs +0.05) and stacking with it. The teacher
  *avoids* wasting compute beyond the frontier; hindsight *recycles* whatever
  still lands there.
- **Ablations:** weight scale is monotone on the toy (AUC 0.805→0.943 at scale
  0.25→8) — expect a knee on real models (over-weighted self-imitation
  entrenches errors); default 1.0 = the natural K=1 group weight. In the
  16-level regime hindsight *partially substitutes* for the teacher
  (uniform+hs 0.970 > advmass-alone 0.961) but they still stack
  (advmass+hs 0.978 best everywhere); the teacher keeps its wall-clock edge on
  real models since it avoids generating doomed rollouts at all.
- **GPU A/B/C (matched 2400 s, frontier_alp teacher) — dense wins, feedback
  dropped:** sparse hindsight ties the no-hindsight baseline (AUC 0.234 vs
  0.233); **dense hindsight** (relabel every failed rollout with prefix ≥ 6,
  cap 16/step, 3.6→16 relabels/step) is the new GPU champion — final 0.258,
  best 0.269, shallow levels lifted across the board (0.99/0.87/0.68/0.45).
  Teacher feedback (C) confirmed V4's pre-registration: AUC tie, worse final,
  visible posterior inflation (p̂ 0.81 vs eval 0.47 at level 2) pushing
  sampling deeper prematurely. **Ship dense hindsight; keep the posterior on
  requested-task evidence only.**
- **Bias caveat:** relabeled goals are selected from the same dead group, so
  samples are coupled and success-conditioned — an auxiliary HER-style term,
  not an unbiased truncated-ML gradient. V1 verifies direction (cosine), not
  magnitude or equality of joint sampling laws. It helps uniformly on the toy;
  on the infinite-data GPU maze its reliable gain is AUC rather than final
  coverage (the three-seed final edge over the plain teacher is small).
- **LLM analogue:** goal/prefix relabeling wherever verifiers admit it —
  sub-goals in multi-step proofs, partial-credit unit tests, reached-state
  goals in agentic tasks.

### M6. Curriculum patches GRPO's inversion (H6) — ❌ REFUTED, direction reversed

- The paper conjectures GRPO's w(p) inversion at p→1 drives distribution
  sharpening. Prediction: a frontier teacher that retires p≈1 prompts removes
  the inversion's regime → less pass@k collapse for `frontier+grpo`.
- **Result (matched wall-clock, GPU maze):** the opposite. frontier+grpo
  collapsed *more* (pass@8 0.332→0.269) than uniform+grpo (0.351→0.312) and
  lost easy-level retention (0.62 vs 0.75 min easy pass). GRPO's inverted
  weight was effectively *maintaining* mastered levels; pointing all its
  updates at the frontier removes that maintenance and sharpens harder.
  MaxRL under the same teacher keeps pass@8 flat-to-up (0.327→0.351).
- **Upshot — sharper than the hypothesis:** GRPO's pass@k collapse is an
  *objective-level* pathology that data-level curricula cannot rescue (they
  amplify it). This strengthens the paper's central claim from the outside:
  to combine a frontier curriculum with group RL you need a likelihood-style
  objective. Teacher and MaxRL aren't just complementary — the teacher
  *requires* MaxRL-style weighting to be safe.

## 2b. Matched wall-clock GPU sweep (complete, seed 0, 2400 s each)

| config | steps | dead/8 | AUC | pass@8 |
|---|---|---|---|---|
| uniform+maxrl | 583 | 5.8 | 0.214 | — |
| uniform+grpo | 527 | 6.0 | 0.216 | 0.312 |
| frontier+grpo | 751 | 5.7 | 0.219 | 0.269 ↓ |
| uniform+maxrl+hindsight | 651 | 5.0 | 0.222 | 0.356 |
| frontier+maxrl | 713 | 3.9 | 0.223 | 0.351 |
| learnability+maxrl | 787 | 3.3 | 0.227 | 0.356 |
| frontier_alp+maxrl | 765 | 3.4 | 0.233 | 0.361 |
| **frontier+maxrl+hindsight** | 694 | 3.8 | **0.234** | 0.356 |

Every teacher/hindsight variant beats both uniform baselines; the winner
combines the teacher with hindsight recycling, and frontier_alp is the best
pure-teacher config (M5's ALP term earns its keep here — level-2 pass 0.62 vs
0.54). GPU hindsight gains are much smaller than CPU (+0.01 vs +0.22 AUC):
in the infinite-data maze regime each relabeled maze is seen once, so the
salvaged signal cannot compound the way it does on a fixed task set — expect
the CPU-like regime on fixed prompt datasets (GSM8K).

**Multi-seed confirmation (seeds 0–2, paired via shared per-seed SFT
warmstart):** both teacher variants beat uniform+maxrl on AUC in **every**
seed (6/6 paired deltas positive; frontier_alp +0.019/+0.004/+0.006,
frontier+hindsight +0.020/+0.002/+0.013), frontier_alp wins final mean-pass
with the tightest spread (0.246 ± 0.002 vs 0.230 ± 0.015). The pass@k
divergence is systematic: GRPO decays coverage in every seed (pass@8 mean
0.308→0.271) while teacher+MaxRL grows it (0.316→0.348) — the paper's
coverage-collapse dynamic reproduced at 1.26M scale, with our addition that
a curriculum widens the gap in both directions.

## 2c. Baseline head-to-head + regime map (V5, matched generation budget)

Against DAPO-style dynamic sampling (redraw until 0<K<N, paying for every
draw), 5 seeds, 3 task-pool regimes — AUC:

| regime | uniform | DAPO | teacher | **teacher+hindsight** |
|---|---|---|---|---|
| easy-heavy | 0.946 | 0.929 | 0.953 | **0.975** |
| balanced | 0.734 | 0.825 | 0.784 | **0.931** |
| frontier-heavy | 0.000 | 0.000 | 0.000 | **0.928** |

The frontier-heavy row is categorical: with max pool pass rate 10⁻⁵, uniform,
DAPO, and the plain teacher all flatline at exactly 0 — reallocating compute
among unlearnable tasks cannot help. Only hindsight *creates* signal: traced,
it relabels dead groups to prefix tasks (inventing the missing curriculum
below the given pool), ignites in-pool learnability within ~400 groups, then
goes nearly silent as natural successes take over. It is a cold-start
igniter, not a permanent crutch. DAPO helps only in the balanced regime and
is subsumed by teacher+hindsight everywhere at equal compute.

## 3. Validated cross-cutting findings

- **Complementarity (H3):** in beyond-frontier-heavy regimes each ingredient alone
  plateaus (uniform+maxrl 0.871, frontier+grpo 0.847) while the combination reaches
  0.961 and the max frontier depth.
- **MaxRL ≈ implicit curriculum (H2):** teacher lifts GRPO by +0.23/+0.42 but MaxRL
  by only +0.01/+0.09 on moderate distributions.
- **Fixed-step comparisons are unfair to teachers:** frontier-level rollouts
  terminate early, so teachers run ~2× more steps per wall-clock second; all GPU
  comparisons now use a `--max-seconds` matched-compute protocol.
- **Curriculum dimensions must respect generalization reach:** maze *size* has a
  hard cliff (0/1024 on 9×9 after 7×7 SFT); goal *distance* within a fixed maze
  transfers smoothly. Curriculum axes need overlapping representations.

## 4. Testbeds

| testbed | scale | purpose |
|---|---|---|
| `curriculum_maxrl/testbed.py` skill-chain | CPU, exact gradients, ~0.2 s/100 steps | fast teacher/estimator ablations; skills shared across chain levels so curricula genuinely matter |
| `curriculum_maxrl/maze_gpu/` | 1.26M-param transformer, A10G | real sampling noise, wall-clock costs, pass@k; mirrors the paper's maze experiment with a difficulty dimension added |
| verl + SmolLM/GSM8K (`verl_integration/`) | 8×GPU node | the paper's data-scarce regime; ready to launch, needs multi-GPU node |

## 5. Roadmap

1. Finish the Isaac Lab Anymal-C rough-terrain simulator pilot and fixed-grid
   evaluation; only ladder step 1 is implemented there.
2. Test deeper supervision/capacity changes for the maze's compounding
   per-step-error ceiling; longer training alone did not solve level 6 pass@1.
3. Validate the existing kernel streaming teacher on a real procedural source;
   it currently has only synthetic continuous-goal evidence.
4. Add per-prompt rollout counts to the verl rollout worker so M2 can be tested
   beyond the CPU allocation model.
5. Re-run adaptive T with the exact Eq. (10) helper before reviving M4.
6. Run SmolLM2-360M + GSM8K `curriculum × {maxrl, grpo}` via
   `verl_integration/smollm_curriculum.sh` — checks whether the dead-prompt gap the
   paper shows in its Fig. 7 closes faster with the teacher; this remains blocked
   on an 8-GPU node.

## 6. Honest limitations

- CPU effect sizes come from a toy with exact gradients; LLM noise (verifier
  errors, nonstationary posteriors) may shrink the teacher's edge — that's exactly
  what the GSM8K run tests.
- Main teacher and dense-hindsight comparisons have three seeds; inference
  efficiency, long-horizon, and wide-model results remain single-seed.
- The discrete teacher assumes a fixed finite prompt set. The kernel streaming
  variant removes that assumption but has only synthetic continuous-goal
  evidence, not a production procedural source.
- `pass@N − p` targets *this group's* signal, not long-horizon transfer; it has no
  notion of prerequisite structure beyond what the floor + posterior drift capture.
