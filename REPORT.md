# Curriculum-MaxRL: Experiment Report and Research Assessment

*Status: 2026-07-23. All numbers reproduce from this repo; provenance for each
table is the JSONL logs in `curriculum_maxrl/` and `curriculum_maxrl/maze_gpu/`.*

---

## 1. Research goal

Starting question: **can a teacher-curriculum improve reinforcement learning
with verifiable rewards, using MaxRL's likelihood objective (arXiv:2602.02710)
as the foundation — and can the combination surpass both plain MaxRL and
standard RL?**

Refined during the project into a sharper claim we set out to prove or refute:

> The right curriculum signal is not an external heuristic; it is computable
> from the estimator's own algebra. And the estimator's blind spot — prompts
> whose rollout groups all fail — defines exactly where the curriculum must be
> supplemented by a second mechanism (hindsight relabeling) that no sampling
> policy can replace.

## 2. The chain from math to algorithm to evidence

Every algorithmic component was derived from a proved identity, then validated
CPU-first (exact gradients), then GPU (1.26M-param transformer, 17×17 mazes).
The chain, component by component:

| Math result (PROOFS.md) | Algorithm component | CPU evidence | GPU evidence |
|---|---|---|---|
| **P0**: Eq. 9/full Eq. 10 have order N; practical dropped-group Algorithm 1 has order N−1 | estimator contract and claim scope | exhaustive finite-N Bernoulli enumeration | production code uses practical Algorithm 1 behavior |
| **P1**: E[Σ\|w\|] = 2(pass@N − pass@1), exact | AdvMass teacher utility | ties hand-tuned ZPD with 0 hyperparams (V2: AUC 0.700 vs 0.688) | teacher beats uniform 6/6 paired seeds; dead groups 5.8→3.4/8 |
| **P2**: peak at p\* ≈ ln N/N, strictly concave | compute-indexed ZPD band; no band tuning | posterior tracks true p within ±0.03 | teacher p̂ matches eval per level; 60% of mass on true frontier |
| **P3**: greedy water-filling optimal for Σ mass | `allocate_rollouts_greedy` | +18% mass vs uniform split | (phase 2 — needs per-prompt n in rollout worker) |
| **P4**: RLOO mass = 2p(1−p) ≡ SFL learnability | unifies curriculum literature w/ estimator algebra | exact MC match | — (interpretive) |
| **P5**: MaxRL mass / RLOO mass → N−1 as p→0 | coefficient-mass advantage on frontier prompts | — | GRPO+teacher collapses (H6 reversed); MaxRL+teacher grows pass@8 every seed |
| **P6**: hindsight bias is the expectation gap between selected-group and fresh joint laws; bounded by their TV distance | dense hindsight relabeling | V1: direction matches fresh groups (cosine 0.956 vs 0.958); mean cosine 1.000, magnitude not tested | dense hindsight = GPU champion (final 0.258, best 0.269) |
| **P7**: posterior lag & floor as staleness bound | decay 0.7, floor 0.1 | V2b: decay 0.7 closes ~19% of oracle gap; V3: floor flat 0–0.4 | defaults shipped in verl module |
| V6 (empirical): chain compounding can favor γ>1 | `power` knob, sample ∝ u^γ | γ=4: AUC 0.782 vs 0.728 | does not transfer to the broad-level maze (0.231 vs 0.236); keep γ=1 |

The two places the chain *broke* were as informative as where it held:

- **Adaptive truncation T** exposed an audit failure: the success coefficient
  is mathematically valid, but the tested wrapper dropped the K=0 score
  control and therefore did not estimate the claimed T objective. That legacy
  variant slightly hurts (AUC 0.698 vs 0.704). The exact Eq. (10) helper is
  now implemented and finite-N verified, but has not been trained.
- **Hindsight→teacher feedback**: V4 pre-registered "safe but redundant, risk
  = optimism inflation"; the GPU run reproduced that signature exactly
  (posterior p̂ 0.81 vs true 0.47 at level 2, worse final). Dropped. The
  posterior must see only requested-task evidence.

Both CPU pre-registrations that made GPU predictions were confirmed — the
toy → GPU pipeline is calibrated, which is the quiet methodological result of
the project.

## 3. Headline findings

### F1. The teacher works, and its mechanism is what the math says it is
At matched wall-clock, every teacher variant beats uniform sampling
(AUC 0.221–0.223 vs 0.211–0.216; 6/6 paired per-seed deltas positive). The
measured mechanism matches P1: dead-group rate drops 5.8→3.4 of 8, steps/sec
rises ~30% (frontier rollouts terminate early), and the sampling mass sits on
the levels where 2(pass@N − pass@1) is largest.

### F2. The curriculum requires the likelihood objective (H6 reversed)
The most instructive refutation: we predicted a frontier teacher would rescue
GRPO from pass@k collapse by retiring the mastered prompts its inverted w(p)
overweights. The opposite: frontier+GRPO collapsed *harder* (pass@8
0.332→0.269) than uniform+GRPO (0.351→0.312), losing easy-level retention,
while MaxRL under the identical teacher *grew* coverage in every seed
(0.316→0.348). GRPO's inversion was maintaining easy prompts; a curriculum
removes that maintenance. **Data-level curricula amplify objective-level
pathologies — a frontier curriculum needs likelihood-style weighting to be
safe.** This strengthens the MaxRL paper's central claim from the outside and
is, we believe, the finding most worth communicating.

### F3. Hindsight breaks the allocator's information ceiling
The oracle teacher (true pass rates — the best possible sampler) reaches CPU
AUC 0.851. Thompson+γ=4+hindsight reaches **0.890**. No contradiction: the
oracle bound constrains *sampling policies*; hindsight changes what a sample
yields, manufacturing verified signal from failures (V1: relabeled directions
align on-structure, mean cosine 1.000). The categorical version: in a
frontier-heavy pool (max p = 10⁻⁵), uniform, DAPO dynamic sampling, and the
plain teacher all flatline at *exactly zero* while teacher+hindsight reaches
0.98 — there was nothing to sample toward; the signal had to be created.
Traced, hindsight invents the curriculum below the given pool, ignites
learnability within ~400 groups, then goes silent. **It is a cold-start
igniter, not a crutch.**

### F4. Sampling concentration is task-graph dependent
On the CPU skill chain, sampling ∝ u^γ with γ≈4 beats γ=1 (0.782 vs 0.728)
and hard top-k. Proportional sampling is an exploration heuristic, not the
one-step signal maximizer: without coverage constraints that maximizer puts
all mass on argmax u. The sharper heuristic does not transfer to the broad-level
GPU maze (AUC 0.231 vs 0.236), so γ=1 remains the GPU/verl default.

### F5. Where everything lands (GPU matched-clock leaderboard, seed 0)

| rank | config | final | best | AUC | pass@8 |
|---|---|---|---|---|---|
| 1 | frontier_alp + maxrl + **dense hindsight** | **0.258** | **0.269** | 0.236 | 0.361 |
| 2 | frontier_alp + maxrl | 0.244 | 0.257 | 0.233 | 0.361 |
| 3 | frontier + maxrl + hindsight | 0.230 | 0.256 | 0.234 | 0.356 |
| … | uniform + maxrl (reference) | 0.225 | 0.233 | 0.214 | — |
| … | uniform + grpo (reference) | 0.230 | 0.237 | 0.216 | 0.312↓ |

## 4. How far are we from the research goal?

**Achieved:**
1. A derived (not heuristic) curriculum — the advantage-mass teacher — with
   proofs, exact validation, and consistent GPU wins. ✅
2. The surpass-normal-RL goal: teacher+MaxRL configurations Pareto-dominate
   GRPO at matched compute on both average performance trend and coverage
   (which GRPO *loses* over training). ✅
3. The surpass-plain-MaxRL goal: +0.016–0.022 final / +0.010–0.018 AUC over
   uniform+MaxRL in the three-seed maze aggregate; and
   categorically (∞×) in frontier-heavy regimes via hindsight. ✅ with the
   caveat that dense hindsight's final edge over the plain teacher is small.
4. A mechanistic account connecting every component to the estimator's
   algebra, with two honest negative results and two confirmed
   pre-registrations. ✅
5. Inference-efficiency evidence in the paper's currency: 1.2×/2.7×/11×
   fewer samples than GRPO at the evaluated frontier targets, with a documented
   mid-level reversal. ✅ single-seed

**Not yet achieved:**
1. **Deep pass@1 remains weak** — a 9600 s run disproved the claim that duration
   alone solves level 6. The measured per-step legality (~0.87) predicts the
   distance-16 stall geometrically; a wider model lifts level-6 coverage@64 to
   0.438 but does not yet establish reliable pass@1.
2. **LLM-scale transfer** — the verl phase-1 sampler is CPU-tested and the
   data-scarce GSM8K regime is precisely where hindsight's fixed-prompt-set
   compounding should shine (the infinite-data maze understates it), but the
   single A10G cannot fit the 8-GPU recipe. Open until a larger node.
3. **P3 (optimal rollout allocation) has no GPU test** — needs per-prompt
   group sizes in the rollout path.
4. **Robotics transfer is incomplete** — Isaac Lab rough-terrain ladder step 1
   is implemented and CPU-tested, but the current simulator pilot has not yet
   produced a completion artifact; later task families are not integrated.

**Assessment: the conceptual goal is met; what remains is scale and task
transfer.** The
project's thesis is no longer "a curriculum might help MaxRL" but a specific,
proved, and doubly-validated statement: *the estimator's algebra defines the
curriculum (P1/P2), the objective determines whether a curriculum is safe at
all (F2), and beyond the allocator's ceiling only signal-creation — not
signal-allocation — makes progress (F3).*

## 5. The most promising next push (and why)

Ranked by expected information:

1. **Complete the Isaac Lab rough-locomotion pilot and fixed-grid readout.**
   This is the nearest external-domain test and gates every later robotic task.
2. **Deep-supervision maze ablation.** Train all verified prefixes or increase
   deep SFT exposure to attack the measured compounding per-step-error ceiling.
3. **Real procedural validation for the streaming teacher.** The kernel
   implementation matches discrete bins on synthetic continuous goals, but has
   not been tested on a production procedural source.
4. **Per-prompt rollout counts in verl.** This is the missing systems path for
   the proved greedy allocation result.
5. **GSM8K 2×2 when hardware allows.** It remains the highest-value LLM transfer
   experiment and is blocked on an 8-GPU node.

## 6. Threats to validity (kept current)

- The main maze ordering has three seeds, but efficiency, long-horizon, and
  wide-model multipliers are single-seed and should not be read as precise
  population estimates.
- CPU effect sizes use exact base-policy gradients and verifier-correct
  relabels, the most favorable case for hindsight. The relabeled goal is still
  selected from the same dead group, so unbiasedness is not established.
  Real verifiers add further mismatch; the maze's small gain vs the toy's
  large one brackets the observed range.
- Matched wall-clock on a shared machine: another process held the GPU for
  part of one window; all compared runs used exclusive windows (checked via
  dead/step + steps-per-second consistency).
- The discrete teacher assumes a finite prompt pool. The kernel streaming
  variant has only synthetic continuous-goal evidence so far.
