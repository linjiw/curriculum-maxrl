# Curriculum-MaxRL: teacher-guided curricula driven by the MaxRL likelihood objective

Working draft. Companion code: `testbed.py`, `estimators.py`, `teachers.py`,
`run_experiment.py` (CPU prototype) and the verl integration sketch at the end.

> **Audit status (2026-07-21).** This file preserves the original design and
> historical experiment names. The corrected estimator conventions and proofs
> are authoritative in `PROOFS.md`: raw `H_N` and always-retained-CV
> `C_N` target order N, while practical drop-both `D_N` targets order N−1.
> Historical `maxrl_frontier` used the nearby legacy score `u_{N+1}`;
> `AdvMassTeacher` uses exact `u_N`.

## 1. What MaxRL is (recap of arXiv:2602.02710)

For a prompt `x` with pass rate `p = p_θ(x)`, maximum likelihood admits the
Maclaurin expansion in failure events:

```
J_ML(x) = log p = -Σ_{k=1..∞} (1-p)^k / k          (eq. 4)
∇J_ML(x) = Σ_{k=1..∞} (1/k) ∇pass@k(x)             (eq. 5)
```

Standard RL (`∇pass@1`) is the first-order truncation. With N rollouts and K
successes, distinguish the raw estimator and the practical centered form:

```
H_N = 1{K>0}(1/K) Σ r_i S_i
C_N = H_N − (1/N)ΣS_i
D_N = 1{K>0}[(1/K)Σr_iS_i − (1/N)ΣS_i]
```

The paper proves `H_N` is unbiased for order N; `C_N` retains that property
only when its control variate remains active at K=0. Practical Algorithm 1
drops both terms at K=0, giving `E[D_N]=∇J_{N-1}`. Thus more rollouts raise
the effective order, but to N−1 for the implementation used in this repo.

The unifying weight-function view (Section 5): all methods have gradient
`E_x[w(p) ∇p]` with

| method | w(p) |
|---|---|
| RL/REINFORCE | 1 |
| GRPO | 1/√(p(1−p)) |
| MaxRL(T) | (1−(1−p)^T)/p |
| ML | 1/p |

## 2. Key observation: MaxRL is an *implicit* curriculum — but only over gradient weights

MaxRL already does at the **gradient level** what curriculum methods do at the
**data level**: it reallocates learning signal toward hard (low-p) prompts,
`w(p)·p = 1−(1−p)^T = pass@T`. But it cannot change *which prompts are in the
batch* or *how many rollouts each gets*. Three gaps remain that an explicit
teacher can close:

1. **Dead prompts.** If `p ≪ 1/N`, all N rollouts fail, K=0, the group is
   dropped — zero gradient (same zero-gradient pathology DAPO dynamic
   sampling attacks). MaxRL's upweighting only kicks in *once you get ≥1
   success*. A teacher must keep the sampled batch inside the band where
   `pass@N` is non-negligible.
2. **Mastered prompts.** At `p→1`, `w·p → 1` but `∇p → 0`; rollouts are
   wasted compute (GRPO is worse here — its w(p) inverts and *upweights*
   p→1). A teacher should retire mastered prompts (with a small replay floor
   against forgetting).
3. **Uniform rollout budget.** The paper uses fixed N per prompt. Group size is
   a per-prompt compute knob: it changes both the live-group probability and
   effective order (N−1 for practical drop-both; N for raw/always-CV forms).

> **Update (see THEORY.md):** the heuristic frontier utility below has been
> superseded by the *derived* advantage-mass utility
> `u(p) = pass@N(p) − p`, which is exactly half the expected total |advantage|
> the practical MaxRL estimator emits on the prompt. The historical heuristic
> is exactly `u_{N+1}`; it is close at the configured N=16/32 but not
> identical at fixed N. The derived form extends
> to an exact myopic fixed-p allocation rule (greedy water-filling on `p(1−p)^N`).

## 3. Historical frontier heuristic and corrected mass utility

The historical teacher began from the population weight-function quantity

```
pass@N(p) = 1 − (1−p)^N
```

and multiplied it by headroom:

```
h_N(p) = (1 − (1−p)^N) · (1 − p) = u_{N+1}(p).
```

- `h_N → 0` when `p → 1` (mastered)
- `h_N → 0` when `p ≪ 1/N` (groups are usually all-fail)
- `h_N` peaks on a "hard but reachable" band
  a ZPD (zone of proximal development) band, but *derived from the estimator*
  rather than hand-tuned like ADARFT's target-difficulty or DAPO's 0<K<N
  filter. As N grows the band automatically widens toward harder prompts —
  the curriculum is compute-indexed like the estimator.

The corrected N-rollout score is
`u_N(p)=pass@N(p)-p=(1-p)pass@(N-1,p)`, exactly half the expected scalar
coefficient mass of `D_N`. This is a proxy/upper-bound factor for gradient
magnitude, not a proof of optimal learning progress.

Estimation uses per-prompt discounted Beta pseudo-counts and Thompson-style
draws. Because the policy moves, these are not exact Bayesian posteriors.

This is teacher–student in the Matiisen et al. sense: student = policy trained
with MaxRL advantages; teacher = non-stationary bandit whose reward is
frontier utility, exploring/exploiting the student's competence boundary.

## 4. Second integration: adaptive rollout allocation (T = N as a curriculum knob)

Given fixed supplied pass rates, a feasible integer budget B, and bounds,
maximize `Σ_i u_{N_i}(p_i)` by assigning each next rollout to the task with
largest marginal `p_i(1-p_i)^{N_i}`. Discrete concavity makes this greedy
rule exact for the one-step coefficient-mass proxy. It is not a theorem about
unknown, moving pass rates or long-horizon learning.

The older `N_i∝1/p̂_i` helper remains for historical experiments. Under
drop-both weights each group targets effective order `N_i−1`; an order
`N_i` claim requires the raw or always-retained-CV estimator.

## 5. Hypotheses to validate

- H1: Under uniform sampling, MaxRL > GRPO on deep skill chains (paper's
  claim, sanity check).
- H2: A teacher helps GRPO more than it helps MaxRL on *moderately* hard
  distributions (MaxRL's implicit weighting already does part of the job).
- H3: On distributions dominated by beyond-frontier tasks (p ≈ 0 for most),
  curriculum+MaxRL ≫ either alone: the teacher fixes the K=0 dead zone that
  MaxRL alone cannot, and practical MaxRL assigns more coefficient mass to
  GRPO.
- H4: MaxRL-frontier teacher (u(p), no hand-tuned band) ≥ ZPD-band teacher
  (hand-tuned [lo,hi]) while having one fewer hyperparameter.
- H5: Adaptive N (budget-preserving) beats fixed N for MaxRL.

## 6. CPU testbed

`SkillChainEnv`: 3 chains × 12 levels, 10 actions/skill; task at level l needs
all of skills 1..l of its chain ⇒ initial p = 10^-l. Exact score functions,
exact pass rates for evaluation (never shown to teacher). This gives real
skill transfer (curriculum matters) with binary verifier rewards (MaxRL
setting) at ~0.2 s per 100 steps.

## 7. verl integration sketch (this repo)

Minimal-diff plan, three pieces:

1. **`CurriculumSampler`** (new, replaces `create_rl_sampler` output when
   `data.curriculum.enable=true`): weighted sampler over dataset indices whose
   weights are updated between iterations from the teacher state. verl already
   supports injecting `train_sampler` into `RayPPOTrainer` (`main_ppo.py:188`).
2. **Teacher state update in `fit()`**: after reward computation each step,
   group `token_level_rewards` by `index` (prompt uid), compute per-prompt
   mean reward, call `teacher.observe(uid, rewards)`; teacher recomputes
   sampler weights `u(Thompson(p̂))`. Persist teacher state in checkpoints
   (it's just two floats per prompt).
3. **(Phase 2) per-prompt `rollout.n`**: needs rollout-side change — vllm
   generation already receives `n` via meta_info; pass a per-sample repeat
   vector instead. Advantage code is already group-size-agnostic
   (`compute_maxrl_outcome_advantage` counts group members by `index`).
   The only caveat: `rearrange`-based estimators assume equal group sizes, so
   keep the maxrl (defaultdict) path.

## 8. Results (skill-chain testbed, 5 seeds, 400 steps, 8 tasks × 16 rollouts/step)

Final mean true pass rate over all 36 tasks (12 levels × 3 chains, initial
p = 10^-level):

| teacher \ estimator | reinforce | rloo | grpo | maxrl |
|---|---|---|---|---|
| uniform | 0.277 | 0.275 | 0.756 | **0.966** |
| zpd band | 0.857 | 0.865 | **0.989** | 0.977 |
| alp | 0.606 | 0.609 | 0.880 | 0.860 |
| maxrl_frontier | 0.673 | 0.664 | **0.989** | 0.979 |

Learning speed (AUC of mean-pass curve; steps until frontier reaches level 12):

| config | AUC | steps→frontier12 |
|---|---|---|
| uniform+grpo | 0.364 | >400 (never) |
| uniform+maxrl | 0.653 | 248 |
| zpd+grpo | 0.655 | 262 |
| zpd+maxrl | 0.688 | 236 |
| maxrl_frontier+grpo | 0.595 | 292 |
| **maxrl_frontier+maxrl** | **0.712** | **206** |
| maxrl_frontier+maxrl+adaptiveN | **0.718** | **200** |

Harder regime (16 levels — most tasks start beyond the frontier, H3):

| config | mean_pass | frontier |
|---|---|---|
| uniform+grpo | 0.423 | 6.8 |
| uniform+maxrl | 0.871 | 14.6 |
| maxrl_frontier+grpo | 0.847 | 14.0 |
| zpd+maxrl | 0.958 | **16.0** |
| maxrl_frontier+maxrl | **0.961** | **16.0** |
| maxrl_frontier+maxrl+adaptiveN | 0.960 | **16.0** |

Hypothesis outcomes:

- **H1 confirmed.** Uniform sampling: MaxRL 0.966 vs GRPO 0.756 (and 0.871 vs
  0.423 in the harder regime). Matches the paper's maze/GSM8K story.
- **H2 confirmed.** A curriculum lifts GRPO by +0.23/+0.42 but MaxRL by only
  +0.01/+0.09 — MaxRL's implicit gradient-level curriculum already covers
  most of what data-level selection buys on this distribution. With a good
  teacher, GRPO ≈ MaxRL on final performance (the teacher keeps everything
  in-band where GRPO's w(p) is fine).
- **H3 confirmed (speed + hard regime).** Curriculum and MaxRL are
  complementary where it counts: maxrl_frontier+maxrl is fastest to the deep
  frontier (206 vs 248 steps uniform+maxrl vs 262 zpd+grpo) and best in the
  beyond-frontier-dominated regime (0.961), where each factor alone plateaus
  (uniform+maxrl 0.871, frontier+grpo 0.847).
- **H4 historical result.** The legacy `h_N=u_{N+1}` frontier teacher
  matches/beats the hand-tuned ZPD band (0.979 vs 0.977 final; AUC 0.712 vs
  0.688). The exact `u_N` teacher is tested separately in `VALIDATION.md`.
- **H5 preliminary.** The historical `1/p̂` adaptive-N heuristic gives a
  small speed gain (AUC 0.718 vs 0.712). It does not test the proved greedy
  allocator, and the earlier T=N interpretation was off by one.
- **Bonus:** ALP (learning-progress) underperforms both pass-rate-band
  teachers here — |Δp̂| is noisy at N=16 rollouts and lags the moving frontier.
- **Estimator ablation:** REINFORCE/RLOO barely move under uniform sampling
  (dead gradients at p ≈ 10^-l) but jump to 0.86 with a ZPD teacher —
  data-level curriculum substitutes for gradient-level reweighting when the
  estimator is weak; the frontier teacher's Thompson band is tuned to MaxRL's
  signal shape and helps GRPO/MaxRL more than REINFORCE.
