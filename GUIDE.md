# Design Guide: approaches, verification status, and roadmap

This is the working map of the project: every approach we proposed, how far each
has been verified, and what's queued next. Detailed derivations live in
`curriculum_maxrl/THEORY.md`; per-experiment tables in `curriculum_maxrl/DESIGN.md`
and `curriculum_maxrl/maze_gpu/EXPERIMENTS.md`.

## 1. Problem framing

MaxRL's estimator (Algorithm 1 of arXiv:2602.02710) normalizes advantages by the
per-prompt mean reward, making it unbiased for the T=N-truncated maximum-likelihood
objective. Its weight function w(p) = (1−(1−p)^T)/p upweights hard prompts — an
*implicit curriculum at the gradient level*. Three gaps remain that only a
*data-level* teacher can close:

1. **Dead prompts** — p ≪ 1/N ⇒ all N rollouts fail ⇒ K=0 ⇒ group dropped, zero
   gradient. No choice of w(p) can put signal where groups die.
2. **Mastered prompts** — p→1 ⇒ ∇p→0; rollouts are wasted compute (GRPO is worse:
   its w(p) *inverts* and upweights p→1, the paper's conjectured cause of pass@k
   collapse).
3. **Uniform rollout budgets** — the paper uses fixed N per prompt, but harder
   prompts need larger N both for signal probability and objective fidelity (T=N).

Our thesis: **the right curriculum signal falls out of the estimator's own algebra**
rather than needing an external heuristic.

## 2. Proposed methods and verification status

### M1. Advantage-mass teacher (core method) — ✅ derived + validated (CPU), 🔄 GPU sweep running

- **Claim (proved, MC-verified 200k trials):** expected total |advantage| a prompt
  receives from a MaxRL group of N rollouts is exactly `2(pass@N − pass@1)`;
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
- **Open:** matched-wall-clock GPU comparison (running: 6 configs × 2400 s).

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

### M4. Adaptive truncation order T (objective curriculum) — ❌ negative result

- **Setup:** the repo's unpublished `c_sub_TN` subset estimator decouples T from N
  (we verified: `c_{N,N}(K)=1/K` recovers Algorithm 1; `E[c·K] = 1−(1−p)^T` to 4
  decimals). Annealed Tᵢ = clip(1/p̂ᵢ, 1, N) per prompt.
- **Result:** slightly *underperforms* fixed T=N (AUC 0.698 vs 0.704 with advmass
  teacher; 0.641 vs 0.653 uniform; 5 seeds). At N=16–32 variance is not the binding
  constraint, so shrinking T only weakens the beneficial hard-prompt upweighting.
- **Status:** documented, deprioritized. Revisit only for very small groups or
  extreme p̂ spreads.

### M5. ALP anti-forgetting term — ⚠️ implemented, weak evidence

- `FrontierALPTeacher` adds an ALP-GMM-style |Δ ema-pass| bonus to re-inject
  regressing levels. On the CPU testbed pure-ALP teachers underperformed
  (|Δp̂| too noisy at N=16 and lags the moving frontier); the additive variant is
  in the matched GPU sweep. The uniform floor already covers most anti-forgetting
  duty in our regimes.

### M6. Curriculum patches GRPO's inversion (H6) — 🔄 hypothesis registered, sweep running

- The paper conjectures GRPO's w(p) inversion at p→1 drives distribution
  sharpening. A frontier teacher retires p≈1 prompts from the batch entirely,
  removing the regime where the inversion applies.
- **Prediction:** `frontier+grpo` shows less pass@8 collapse than `uniform+grpo`
  at matched wall-clock. If confirmed: one posterior fixes GRPO's easy-prompt
  pathology and MaxRL's dead-group pathology simultaneously.

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

1. **Now:** finish matched-clock GPU sweep → test H6/H7 → update EXPERIMENTS.md.
2. Multi-seed (≥3) confirmation of the GPU winner configs.
3. SmolLM2-360M + GSM8K: `curriculum × {maxrl, grpo}` 2×2 via
   `verl_integration/smollm_curriculum.sh` — checks whether the dead-prompt gap the
   paper shows in its Fig. 7 closes faster with the teacher.
4. Phase-2 verl: per-prompt rollout budgets (M2) in the rollout worker.
5. Optional: SEC-style empirical-|advantage| teacher as a baseline against the
   posterior-based advmass teacher (same signal, different estimation path).

## 6. Honest limitations

- CPU effect sizes come from a toy with exact gradients; LLM noise (verifier
  errors, nonstationary posteriors) may shrink the teacher's edge — that's exactly
  what the GSM8K run tests.
- Matched GPU results so far are single-seed.
- The teacher assumes a fixed finite prompt set (Beta posterior per row index);
  streaming/procedural prompt sources need a parametric difficulty model
  (ALP-GMM-style) instead.
- `pass@N − p` targets *this group's* signal, not long-horizon transfer; it has no
  notion of prerequisite structure beyond what the floor + posterior drift capture.
