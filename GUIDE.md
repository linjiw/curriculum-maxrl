# Design Guide: approaches, verification status, and roadmap

This is the working map of the project: every approach we proposed, how far each
has been verified, and what's queued next. Detailed derivations live in
`curriculum_maxrl/THEORY.md`; per-experiment tables in `curriculum_maxrl/DESIGN.md`
and `curriculum_maxrl/maze_gpu/EXPERIMENTS.md`.

> **Audit status (2026-07-21).** The proof and CPU/Gym code have been
> corrected. GPU tables below are historical: they used legacy `u_{N+1}`,
> mixed K=0 with K=N in zero-weight counters, used path length rather than BFS
> depth for hindsight, trained every level with the deepest move budget, scaled
> dense hindsight with relabel count, and logged sparse counter snapshots. They are
> hypothesis-generating until the corrected `advmass` factorial is rerun.

## 1. Problem framing

MaxRL's raw success-average estimator is unbiased for the order-N truncated
maximum-likelihood objective. Its always-retained score control variate keeps
that property. Practical Algorithm 1 drops both terms at K=0 and instead
targets order N−1. The population weight function upweights hard prompts—an
implicit curriculum at the gradient level. Three gaps remain that a
data-level teacher can address:

1. **Dead prompts** — p ≪ 1/N ⇒ all N rollouts fail ⇒ K=0 ⇒ group dropped, zero
   gradient. No choice of w(p) can put signal where groups die.
2. **Mastered prompts** — p→1 ⇒ ∇p→0; rollouts are wasted compute (GRPO is worse:
   its w(p) *inverts* and upweights p→1, the paper's conjectured cause of pass@k
   collapse).
3. **Uniform rollout budgets** — the paper uses fixed N per prompt, but harder
   prompts may benefit from larger N for live-group probability and effective
   order (N−1 practical; N raw/always-CV).

Our thesis: **the right curriculum signal falls out of the estimator's own algebra**
rather than needing an external heuristic.

## 2. Proposed methods and verification status

### M1. Coefficient-mass teacher (core) — ✅ proof + CPU/Gym, ⏳ corrected GPU

- **Claim (proved, MC-verified 200k trials):** expected total |advantage| a prompt
  receives from a MaxRL group of N rollouts is exactly `2(pass@N − pass@1)`;
  peaks at p* ≈ ln(N)/N.
- **Method:** discounted Beta pseudo-counts over each prompt's pass rate;
  Thompson-style draw p; prioritize prompts by `(1−(1−p)^N) − p`;
  uniform floor (default 0.1) for coverage/anti-forgetting.
- **Verification:**
  - Formula: Monte-Carlo match to 3 decimals across p ∈ [0.005, 0.95] (THEORY.md §2, §5).
  - Corrected exact-`u_N` CPU skill-chain (5 seeds): AUC 0.700 versus 0.650
    uniform. The older 0.704–0.712 versus 0.688 ZPD comparison used the legacy
    `u_{N+1}` score and is retained only as historical evidence.
  - Historical GPU diagnostics are not exact-`u_N` validation; see audit note.
- **Open:** corrected matched-wall-clock GPU comparison with cumulative K=0/K=N
  accounting and a common replay floor.

### M2. Greedy rollout allocation — ✅ derived + validated (CPU-level)

- **Claim (proved):** for fixed known pass rates, feasible integer bounds, and
  a fixed one-step budget, maximizing half-mass
  Σᵢ[(1−(1−pᵢ)^{Nᵢ}) − pᵢ] (equivalently total mass after ×2) is discretely
  concave ⇒ greedy water-filling is exact. Half-mass marginal for the
  (N+1)-th rollout on prompt i is `pᵢ(1−pᵢ)^N`—the probability
  that rollout is the group's **first success**.
- **Archived exploratory check (driver/output not retained):** +18% total
  advantage mass vs uniform split on one mixed-difficulty batch and a small
  checkpoint-mean gain over the older 1/p̂ heuristic. This is not part of the
  reproducible evidence set; the exact greedy theorem is.
- **Open:** requires per-prompt `rollout.n` support in the verl rollout worker
  (phase-2 integration); untested at LLM scale.

### M3. Exact learnability equivalences (analysis) — ✅ proved

- RLOO's expected scalar coefficient mass is `2p(1−p)`, proportional to
  SFL's learnability score. The score matches; the estimator and curriculum
  objective are otherwise distinct.
- GRPO's *realized* finite-sample mass on hard prompts is ~2× below its population
  w(p) (at p=0.01, N=32: 0.100 vs 0.199) because 72% of groups are all-fail.
  The paper's population-level weight curves describe the infinite-sample limit only.
- Connection to SEC (Chen et al. 2025c): SEC drives a curriculum bandit with
  empirical |advantage| as reward; our formulas are the exact expectations of
  SEC's signal. The pseudo-count teacher is a model-based estimate of expected
  SEC-style coefficient mass, not an oracle.

### M4. Adaptive truncation order T — ⚠️ historical test inconclusive

- **Setup:** the repo's unpublished `c_sub_TN` subset estimator decouples T from N
  (we verified: `c_{N,N}(K)=1/K` recovers Algorithm 1; `E[c·K] = 1−(1−p)^T` to 4
  decimals). Annealed Tᵢ = clip(1/p̂ᵢ, 1, N) per prompt.
- **Result:** the historical implementation slightly underperforms its fixed
  setting, but it dropped the control variate outcome-dependently and has
  population weight `w_T(p)-(1-p)^(N-1)`. It therefore did not compare the
  claimed order-T objectives. A corrected always-CV rerun is required.

### M5. ALP anti-forgetting term — ⚠️ implemented, weak evidence

- `FrontierALPTeacher` adds an ALP-GMM-style |Δ ema-pass| bonus to re-inject
  regressing levels. On the CPU testbed pure-ALP teachers underperformed
  (|Δp̂| too noisy at N=16 and lags the moving frontier); the additive variant is
  in the matched GPU sweep. The uniform floor already covers most anti-forgetting
  duty in our regimes.

### M7. Hindsight relabeling for dead groups — ✅ validated (CPU), 🔄 GPU sweep queued

- **Idea:** MaxRL's Theorem 1 expresses the ML gradient through a
  success-conditioned score. Practical centered updates also score failures in
  live groups, but discard K=0 groups. HER's move is
  the complement: a failed trajectory is a success **for the goal it actually
  reached**. Where task structure and a verifier admit relabeling, eligible
  dead groups may yield an auxiliary update for an easier related task at zero
  extra generation cost. Centered relabeling still needs a nondegenerate
  rewritten group; success-only relabeling has a different exactness condition.
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
- **Archived exploratory ablations (driver/output not retained):** weight
  scale was monotone on the toy (checkpoint mean 0.805→0.943 at scale
  0.25→8) — expect a knee on real models (over-weighted self-imitation
  entrenches errors); default 1.0 = the natural unscaled multiplier. In the
  16-level regime hindsight *partially substitutes* for the teacher
  (uniform+hs 0.970 > advmass-alone 0.961) but they still stack
  (advmass+hs 0.978 best everywhere); the teacher keeps its wall-clock edge on
  real models since it avoids generating doomed rollouts at all.
- **Historical GPU A/B/C (matched 2400 s, frontier_alp teacher):** sparse
  hindsight ties the no-hindsight baseline (AUC 0.234 vs
  0.233); **dense hindsight** (relabel every failed rollout with prefix ≥ 6,
  cap 16/step, 3.6→16 relabels/step) had the best historical point estimates —
  final 0.258, best 0.269. Its loss scaled with relabel count, however, so the
  comparison confounds coverage and update magnitude. Teacher feedback (C)
  had an AUC tie, worse final,
  visible pseudo-count inflation (p̂ 0.81 vs eval 0.47 at level 2) pushing
  sampling deeper prematurely. These results are provisional because the old
  relabeler confused legal path length with BFS depth and training budgets did
  not match evaluation. Keep teacher state on requested-task evidence only;
  rerun the relabeling comparison.
- **Bias caveat:** relabeled groups are conditioned on the achieved outcome —
  an auxiliary HER-style term, not an unbiased truncated-ML gradient. Helps
  uniformly on the toy; GPU maze version (goal ← deepest cell legally reached,
  `--hindsight` in `maze_gpu/train.py`) is queued behind sweep 1.
- **LLM analogue:** goal/prefix relabeling wherever verifiers admit it —
  sub-goals in multi-step proofs, partial-credit unit tests, reached-state
  goals in agentic tasks.

### M6. Curriculum patches GRPO's inversion (H6) — historical refutation

- The paper conjectures GRPO's w(p) inversion at p→1 drives distribution
  sharpening. Prediction: a frontier teacher that retires p≈1 prompts removes
  the inversion's regime → less pass@k collapse for `frontier+grpo`.
- **Historical result (matched wall-clock, GPU maze):** the opposite. frontier+grpo
  collapsed *more* (pass@8 0.332→0.269) than uniform+grpo (0.351→0.312) and
  lost easy-level retention (0.62 vs 0.75 min easy pass). GRPO's inverted
  One interpretation is that easy-level updates helped maintain coverage,
  but the audited runs do not isolate that mechanism. MaxRL under the same
  teacher kept pass@8 flat-to-up (0.327→0.351).
- **Historical hypothesis:** in these audited runs the frontier curriculum
  coincided with worse GRPO pass@k retention and better MaxRL retention. This
  suggests an objective-level interaction, but the three-seed provisional
  evidence does not establish that data curricula generally amplify GRPO or
  require likelihood-style weighting for safety.

## 2b. Historical matched-wall-clock GPU sweep (provisional)

| config | steps | old zero-weight snapshot/8 | legacy unanchored step-AUC | pass@8 |
|---|---|---|---|---|
| uniform+maxrl | 583 | 5.8 | 0.214 | — |
| uniform+grpo | 527 | 6.0 | 0.216 | 0.312 |
| frontier+grpo | 751 | 5.7 | 0.219 | 0.269 ↓ |
| uniform+maxrl+hindsight | 651 | 5.0 | 0.222 | 0.356 |
| frontier+maxrl | 713 | 3.9 | 0.223 | 0.351 |
| learnability+maxrl | 787 | 3.3 | 0.227 | 0.356 |
| frontier_alp+maxrl | 765 | 3.4 | 0.233 | 0.361 |
| **frontier+maxrl+hindsight** | 694 | 3.8 | **0.234** | 0.356 |

In these historical logs, every teacher/hindsight variant has higher legacy
step-AUC than both uniform baselines in seed 0. The audited confounds prevent
attributing that ordering to the teacher or hindsight. The old GPU hindsight
delta is much smaller than CPU (+0.01 vs +0.22 AUC):
in the infinite-data maze regime each relabeled maze is seen once, so the
salvaged signal cannot compound the way it does on a fixed task set — expect
the CPU-like regime on fixed prompt datasets (GSM8K).

**Historical multi-seed round (seeds 0–2, shared per-seed SFT weights but not
common post-SFT RNG streams):** both teacher variants had positive legacy step-AUC deltas in
each seed (frontier_alp +0.019/+0.004/+0.006,
frontier+hindsight +0.020/+0.002/+0.013), frontier_alp has higher final mean-pass
with the tightest spread (0.246 ± 0.002 vs 0.230 ± 0.015). GRPO coverage
declined and teacher+MaxRL coverage grew in these three runs, but no formal
test and the audit confounds rule out a systematic-mechanism claim.

## 2c. Baseline head-to-head + regime map (V5, matched generation budget)

Against DAPO-style dynamic sampling (redraw until 0<K<N, paying for every
draw), 5 seeds, 3 task-pool regimes — AUC:

| regime | uniform | DAPO | teacher | **teacher+hindsight** |
|---|---|---|---|---|
| easy-heavy | 0.880 | 0.880 | 0.887 | **0.912** |
| balanced | 0.645 | 0.645 | 0.699 | **0.863** |
| frontier-heavy | 0.000 | 0.000 | 0.000 | **0.860** |

The frontier-heavy row is categorical: with max pool pass rate 10⁻⁵, uniform,
DAPO, and the plain teacher all flatline at exactly 0 — reallocating compute
among unlearnable tasks cannot help. Only hindsight *creates* signal: traced,
it relabels dead groups to prefix tasks (inventing the missing curriculum
below the given pool), ignites in-pool learnability within ~400 groups, then
goes nearly silent as natural successes take over. It is a cold-start
igniter, not a permanent crutch. After correcting irregular evaluation
timestamps, DAPO equals uniform exactly in this sequential, total-generation-
matched simulator: both process the same uniform sample stream and update on
the same live groups. The earlier apparent balanced-regime gain was an AUC
measurement artifact.

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

1. **Now:** rerun GPU uniform vs exact `advmass` vs legacy frontier with a
   common floor, paired SFT state, cumulative K=0/K=N logging, and no hindsight.
2. Add corrected BFS-depth hindsight only after the teacher effect is clear.
3. Run ≥10 paired Gym seeds and the N/group-size frontier-shift ablation.
4. SmolLM2-360M + GSM8K: `curriculum × {maxrl, grpo}` 2×2 via
   `verl_integration/smollm_curriculum.sh` — checks whether the dead-prompt gap the
   paper shows in its Fig. 7 closes faster with the teacher.
5. Phase-2 verl: per-prompt rollout budgets (M2) in the rollout worker.
6. Optional: SEC-style empirical-|advantage| teacher as a baseline against the
   pseudo-count advmass teacher (same signal, different estimation path).

## 6. Honest limitations

- CPU effect sizes come from a toy with exact gradients; LLM noise (verifier
  errors, nonstationary pass-rate estimates) may shrink the teacher's edge—that is
  what the GSM8K run tests.
- Corrected GPU results do not yet exist; historical tables have the audit
  confounds stated above.
- The teacher assumes a fixed finite prompt set (pseudo-counts per row index);
  streaming/procedural prompt sources need a parametric difficulty model
  (ALP-GMM-style) instead.
- `pass@N − p` targets *this group's* signal, not long-horizon transfer; it has no
  notion of prerequisite structure beyond what the floor + pseudo-count drift capture.
