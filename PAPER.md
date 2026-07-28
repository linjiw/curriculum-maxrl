# The Estimator Is the Curriculum
### Frontier Sampling and Failure Recycling for Likelihood-Based RL

*Working draft v0.9 (2026-07-27). All numbers reproduce from this repository;
experimental tables in REPORT.md and GSM8K_ANALYSIS.md; proofs in
curriculum_maxrl/PROOFS.md; graded claim inventory in EVIDENCE.md.*

**TL;DR — Curricula and likelihood-based RL are not two ideas but one: the
estimator's own algebra defines the optimal curriculum, and what the
estimator throws away defines what a curriculum cannot do — until you
recycle it.**

---

## Abstract

Reinforcement learning with verifiable rewards spends most of its budget
generating rollouts, and on hard task pools most of that budget buys
nothing: under uniform prompt sampling, we measure that **65–75% of rollout
groups produce zero learning signal** — every rollout fails, and
success-conditioned estimators drop the group. The emerging fix is a
difficulty curriculum, but current curricula are heuristics bolted on from
outside: bandit scores, hand-set difficulty bands, filtering rules. We show
the curriculum is already inside the estimator. For MaxRL-style
success-conditioned advantages, the expected learning signal a prompt emits
from a group of N rollouts has a closed form, **E[Σ|w|] = 2·(pass@N −
pass@1)** — a compute-indexed zone-of-proximal-development functional,
maximal on prompts solvable within N attempts but not within one, peaking
at pass rate ≈ ln N/N. Sampling prompts by this quantity (a decayed
Thompson posterior makes it practical) turns the rollout budget N you
already chose into the curriculum knob, with no new difficulty
hyperparameters; published learnability curricula fall out as its N=2
slice. The same algebra draws a hard boundary: no sampling rule can rescue
an all-fail prompt, because allocation only redistributes signal that
exists. We therefore add a second, complementary mechanism — **failure
recycling**: a failed trajectory is a verified success for the goal it
actually reached, and relabeling dead groups (with the prompt's goal
conditioning rewritten) provably yields the maximum-likelihood gradient of
the relabeled task, exact when conditional laws match. On testbeds spanning
exact-gradient chains, a 1.26M-parameter maze transformer, Gymnasium
control, and a 360M LLM on GSM8K, the combined schedule beats a
true-pass-rate **oracle** allocator (0.890 vs 0.851 AUC), turns a
frontier-heavy regime from unlearnable to solved (0 → 0.98 at equal
compute), grows pass@k coverage where GRPO's collapses, and needs up to
11× fewer inference samples to reach target coverage. A pre-registered
LLM-scale 2×2 confirms the safety half of the claim: the identical
curriculum that trains stably under MaxRL *degrades* GRPO — curricula
amplify objective-level pathologies, so the estimator underneath decides
whether a curriculum is safe at all.

---

## 1. Introduction

Post-training language models with verifiable rewards has a cost structure
unlike ordinary supervised learning: the data is free, but *rollouts* are
not. Every optimization step is preceded by sampling N completions per
prompt, and generation dominates wall-clock (86% of step time in our LLM
runs). What that compute buys depends entirely on where it lands. On a
prompt the model has mastered, N successes carry no contrast and no
gradient. On a prompt beyond the model's reach, N failures carry no
successes — and for the success-conditioned estimators of likelihood-based
RL, the group is dropped entirely. We measure the combined waste at 65–75%
of groups under uniform sampling. The field's response is difficulty
curricula — but the current generation derives its sampling rule from
outside the learner: UCB bandits over difficulty buckets (DUMP),
target-difficulty controllers (AdaRFT), category bandits (SEC), rejection
rules (learnability sampling), or dead-prompt filtering (DAPO's dynamic
sampling, GRESO). Each adds machinery and hyperparameters to estimate, in
effect, *where learning is possible right now*.

Our starting observation is that likelihood-based RL already computes
this. MaxRL (Tajwar, Zeng, et al., 2026) showed that standard RL on binary
rewards optimizes a first-order approximation of maximum likelihood, and
that normalizing advantages by the group's success count K instead of its
size N recovers a truncated-ML objective. Reading their weight-function
view raises a question with a non-obvious answer: *if the objective
already pours gradient into hard prompts, is there anything left for a
curriculum to do?* The answer is sharply yes, for a structural reason:
**weights act on prompts after they are sampled, and only when at least
one rollout succeeds.** However the weight function is shaped, it cannot
rescue a group that came back all-fail — dropping that group is precisely
what makes the estimator unbiased — and it cannot un-spend the compute
burned re-confirming mastered prompts. The estimator's blind spots define
the curriculum's job description.

Taking that sentence literally produces the paper. Our contributions:

1. **The curriculum is a theorem, not a heuristic (§3).** The expected
   total advantage magnitude a prompt receives from a group of N rollouts
   under success-conditioned weights is exactly 2·(pass@N − pass@1)
   (Proposition 1, MC-verified). This is a compute-indexed ZPD functional:
   zero at both mastery and unreachability, peaked at p* ≈ ln N/N.
   Sampling by it requires no difficulty band, no bucket boundaries, no
   threshold — the group size N *is* the curriculum knob. Three
   corollaries unify prior art: RLOO's expected signal is exactly 2p(1−p),
   the published "learnability" objective (its N=2 slice); MaxRL
   concentrates ≈(N−1)× more signal than RLOO on frontier prompts, which
   is why a frontier curriculum is safe with MaxRL specifically; and
   optimal rollout allocation is greedy water-filling on p(1−p)ᴺ, the
   probability the next rollout is a group's first success.
2. **A practical teacher with the derived utility (§4).** FrontierMax: a
   decayed Beta posterior per prompt, Thompson sampling proportional to
   u(p̃) = (1−(1−p̃)ᴺ) − p̃, a uniform exploration floor, and unmodified
   MaxRL advantages underneath — every unbiasedness property of the base
   estimator carries over untouched.
3. **Failure recycling with an exactness guarantee (§5).** A failed
   trajectory is a verified success for the sub-goal it reached.
   Relabeling dead groups to achieved sub-goals and applying the same
   success-conditioned weights yields the ML gradient of the relabeled
   task under a shifted conditional law (Proposition 6) — exact when the
   laws match, and measured indistinguishable from fresh unbiased groups
   (per-group cosine 0.956 vs 0.958; mean gradient cosine 1.000). Two
   contracts keep it exact in practice, and violating the second
   (conditioning rewrite) makes hindsight actively harmful — we measured
   the cost.
4. **A three-channel account with a safety rule (§6–7).** Allocation (the
   teacher) is bounded by an oracle ceiling and worth +0.05–0.08 AUC;
   creation (recycling) breaks that ceiling and owns the frontier-heavy
   regime; and the objective underneath decides whether either is safe —
   the identical teacher grows coverage under MaxRL in every seed and
   *amplifies* GRPO's pass@k collapse. Confirmed at LLM scale on a
   pre-registered prediction (§7.5).
5. **An honest evidence discipline.** Predictions pre-registered before
   data (GSM8K 2×2, IsaacLab); negative results documented with diagnoses
   (γ-concentration non-transfer — predicted in advance by an ODE model;
   adaptive truncation order; learning-progress teachers; one retracted
   early claim); every quantitative claim in this draft traced to a log or
   proof by a two-round adversarial audit.

## 2. Background: MaxRL in three lines

For binary rewards, maximum likelihood maximizes E[log p_θ(success|x)].
MaxRL expands log p as a Maclaurin series in (1−p) and truncates at order
T, giving a compute-indexed family J^(T) interpolating REINFORCE (T=1) to
exact ML (T→∞); the practical estimator sets per-rollout advantages
w_i = r_i/K − 1/N (successes) with all-fail groups dropped, which is
unbiased for the T = N−1 objective. Its weight function w(p) =
(1−(1−p)ᵀ)/p grows as p→0: the objective is an *implicit, gradient-level*
curriculum. Our work is the data-level complement: the same algebra, read
as a sampling rule plus a recycling rule.

## 3. The estimator is the curriculum

**Proposition 1 (advantage mass; MC-verified).** *For a prompt with pass
rate p and N i.i.d. rollouts under the practical MaxRL weights,*

    E[Σᵢ|wᵢ|] = 2·(pass@N(p) − pass@1(p)) = 2·((1−(1−p)ᴺ) − p).

**Interpretation.** The estimator's expected learning signal on a prompt
is twice the probability that the prompt is *solvable within N attempts
but not within one*. That sentence is a curriculum: zero on mastered
prompts (p→1), zero on unreachable ones (p→0), maximal at
p* = 1 − N^(−1/(N−1)) ≈ ln N/N — a zone of proximal development whose
center, width, and compute-scaling are all set by the one parameter you
already chose, the rollout budget N. Raise N and the band walks toward
harder prompts automatically.

**Corollary (one identity, three literatures).**

| prior art | is exactly | via |
|---|---|---|
| learnability curricula p(1−p) (SFL, LILO) | the N=2 slice of u_N | Prop. 4: RLOO mass = 2p(1−p) exactly |
| DAPO dynamic-sampling / GRESO filtering | the avoidance shadow of u_N (cull where u ≈ 0) | u → 0 at both ends |
| HER-style relabeling | the creation complement (manufacture K>0 where u = 0 identically) | §5 |

**Proposition 5 (why frontier curricula need likelihood weighting).** As
p→0, MaxRL's expected signal exceeds RLOO's by a factor → N−1.
**Interpretation:** concentrating sampling on the frontier only helps if
the estimator can extract signal there. MaxRL's (N−1)× frontier
amplification is the finite-sample mechanism; GRPO's variance-normalized
weights invert the profile — which §7 shows becoming an active failure
under a curriculum.

**Proposition 3 (allocation).** The rollout-budget allocation maximizing
total expected signal is greedy water-filling on the marginal p(1−p)ᴺ —
the probability that the next rollout is a group's first success.

## 4. FrontierMax: the schedule

A decayed Beta posterior (decay 0.7) tracks each prompt's pass rate from
observed group outcomes only — never from relabeled successes (§5's
hygiene rule; violating it inflates the posterior, p̂ 0.81 vs eval 0.47).
Thompson sampling draws p̃ and samples prompts ∝ u(p̃)^γ with a 10% uniform
floor; γ≈4 when tasks share skills (learning compounds — validated on
chains, and its non-transfer to broad pools was predicted in advance by a
compounding ODE model), γ=1 otherwise. Live groups train with unmodified
MaxRL advantages. Honest knob inventory: decay, floor, and γ exist, with
validated defaults; what is *derived* rather than tuned is the difficulty
band itself — location, width, and N-scaling — which is exactly where
competing curricula spend their hyperparameters.

## 5. Failure recycling

**Proposition 6 (hindsight exactness).** *Relabeling a dead group to the
sub-goal its trajectories actually achieved, rewriting the goal
conditioning, and applying the same success-conditioned weights yields the
ML gradient of the relabeled task under the conditional law shifted by the
original sampling; the two coincide exactly when the conditional laws
match.* **Measured:** per-group cosine to the true relabeled-task gradient
0.956, vs 0.958 for fresh unbiased groups; the mean relabeled gradient
reaches cosine 1.000.

Two contracts keep practice on the proof's side: **(1) exactness** — a
relabeled success must be a true success of the relabeled task under the
environment's own verifier (never an LLM judge); **(2) conditioning** — if
trajectories embed the goal (goal tokens in a prompt, desired-goal
observations), the conditioning must be rewritten to the achieved goal.
Skipping (2) turns the exact gradient into noise: on our gridworld,
hindsight *without* the rewrite scores below teacher-only (0.600 < 0.658);
with it, it leads (0.703).

Recycling is the only mechanism here that *creates* signal rather than
reallocating it, and its value is regime-dependent in a way we can state
precisely: the gain is proportional to how much a relabeled skill can
compound — **+0.22 AUC on fixed task pools** (skills recur), **+0.01 on
one-shot task streams** (they don't). Task-set fixedness is the single
most important regime variable we found.

## 6. Three channels, one safety rule

The **teacher** avoids waste — bounded by an oracle ceiling (a perfect
allocator collects only 0.4% more advantage mass than our posterior).
**Recycling** creates signal — the only channel that breaks that ceiling.
The **objective** underneath decides whether either is safe. One line:
*the teacher allocates, hindsight creates, the objective decides whether
either is safe.*

## 7. Experiments — an escalating ladder

Each rung isolates the channels it can test cleanly; predictions at the
LLM rung were pre-registered and committed before any cell finished.

**7.1 Exact-gradient skill chains (CPU, 5 seeds).** Uniform 0.650 →
teacher 0.728 → full stack **0.890 ± 0.002** — above the true-pass-rate
oracle allocator (0.851) and above a sharper γ=4 oracle (0.884); artifact
`frontier_rl/examples/v7_oracle_result.json`.
**Takeaway: allocation saturates; creation breaks the ceiling.**

**7.2 Frontier-heavy regime (max pool pass rate 10⁻⁵).** Uniform, DAPO,
and the plain teacher score **exactly 0.00**; teacher+recycling reaches
**0.98 final (0.93 AUC)** — relabeling invents the curriculum below the
pool, ignites within ~400 groups, then goes silent.
**Takeaway: where there is nothing to sample toward, only signal creation
works — categorically, not marginally.**

**7.3 Maze transformer at matched wall-clock (1.26M params, 13
goal-distance levels, ~30 runs, 3 seeds).** Champion (teacher + dense
recycling) 0.252±0.005 final / 0.229±0.009 AUC vs uniform 0.230±0.015 /
0.211±0.011; paired deltas positive 6/6 seeds; 22–35% more optimization
steps per GPU-hour. The safety result (H6): the identical teacher grows
pass@8 under MaxRL in every seed (0.316→0.348) while GRPO's coverage
decays in every seed (0.308→0.271); the teacher-arm run amplifies the
collapse (0.332→0.269; single-seed arm). Inference currency: our
checkpoint needs 1.2×/2.7×/**11×** fewer samples than GRPO's to reach
target coverage at levels 2/3/5 (honest 0.5× reversal at level 4) — the
base paper's 2.3–19.2× pattern reproduced at 1.26M scale with the teacher
on top, and GRPO's coverage curves *flatten* at large k (saturating at
0.88/0.56 where ours reach 1.00/0.62).
**Takeaway: the gains survive real gradients; the safety warning is real;
coverage is the currency that sees both.**

**7.4 Gymnasium control (MountainCar).** Training on the flag alone scores 0.000 (the classic exploration
wall). The corrected 10-seed transition-matched study (paired bootstrap,
from the audited branch; per-seed curves committed there): flag pass
uniform 0.058±0.079 → teacher (γ=4, exact mass) 0.664±0.232 → full stack
**0.848±0.058**; the same curriculum with *per-bin* policy parameters
stays at 0.000.
The instructive failure stands: curricula operate through shared
parameters, or not at all.
**Takeaway: curricula operate through shared parameters, or not at all —
the first thing to check when a curriculum "doesn't work."**

**7.5 LLM scale — GSM8K 2×2, pre-registered (SmolLM2-360M, N=16, one
A10G).** Axes {MaxRL, GRPO} × {teacher, uniform}; predictions committed
before any cell finished. **P-G2, the riskiest prediction, confirmed:**
GRPO+teacher is the only cell that regresses in its second half (val
accuracy .096→.093, pass@4 .193→.181) while plain GRPO climbs
monotonically (.078→.105→.120) — the H6 reversal at LLM scale, with the
teacher verifiably steering (sampled-dead fraction driven to 0.48 vs the
~0.65 population rate). MaxRL+teacher trains stably to its best value
(.066→.102). Beyond the predictions, the run cleanly separates the
teacher's two sub-channels: its posterior *learns* real difficulty from
~1 visit per prompt (ρ = −0.17 against independent 7B-model solve-rate
annotations, p ≈ 10⁻¹⁷), but its allocation is *posterior-starved* — 3,200
draws over 7,473 prompts leave Thompson sampling near-uniform. The teacher
knows more than it can act on. And the GRPO degradation is not
token-entropy collapse (GRPO+teacher retains *more* entropy than GRPO):
the damage is distributional, visible only in coverage. Single seed, small
model, deltas in points not leaps — the pre-registered outcomes were
orderings and signs, and both landed.
**Takeaway: the safety half of the thesis transfers to LLM RLVR; the
allocation half needs tiers or longer runs — which is exactly how the next
experiment is designed.**

**Next rung (staged, review-hardened): Countdown 2×2×2** — the first
exact-verifier hindsight experiment in RLVR: a failed equation still
evaluates to some value v; relabel the target to v and the same verifier
certifies a true success. Pre-registered prediction: recycling ignites the
operand tiers that stay at zero for every recycling-off cell (the §7.2
pattern at LLM scale).

## 8. Related work

Difficulty-adaptive RLVR sampling is bucket- or scalar-level and
heuristic-scored: DUMP (UCB over difficulty buckets, |advantage| reward),
AdaRFT (scalar target-difficulty controller over precomputed labels), SEC
(category bandit), threshold curricula (reasoning-gym §5). LILO is the
closest per-prompt method — rejection sampling by p̂(1−p̂), which
Proposition 4 identifies as the N=2 slice of our derived utility. DAPO's
dynamic sampling and GRESO cull dead prompts — avoidance without
allocation or creation; the discards still cost GPU-hours. PKPO and
Pass@k-training modify the objective toward pass@k but keep uniform
sampling — the complement of our teacher, with no recycling. Hindsight for
LLMs (AgentHER, HSL) relabels agent trajectories with an LLM judge. The
nearest neighbor is concurrent: LfH (Xu et al., 2026) brings hindsight
relabeling into GRPO for VLA post-training — a VLM proposes one shared
hindsight instruction per failed group and rescores the group under it,
with a hindsight importance correction; 5× sample efficiency on
LIBERO-PRO. LfH's relabeler is a VLM judge (their stated limitation:
relabel noise), and its evaluation never measures pass@k — precisely the
currency where we find recycling's hidden cost. Exact-verifier relabeling
with a correctness guarantee, and the coverage accounting of recycling,
remain, to our knowledge, ours. No prior work (i) derives the sampling utility from the
estimator's own expected signal, (ii) couples it with success-conditioned
weights it provably matches, and (iii) adds a signal-creation channel with
an exactness guarantee.

## 9. Limitations and honest negatives

Deep frontiers at fixed budget remain uncrossed on the maze, and a 4×
duration run *refuted* our own duration hypothesis (level 6 stays ≈0.01;
the stall is a per-step-competence ceiling, q≈0.87 capping geometric
reach) — depth needs capacity or deeper warmstarts, not more schedule.
γ-concentration did not transfer from chains to the maze (0.231 vs 0.236
AUC) — predicted in advance by the compounding ODE model. Adaptive
truncation order underperformed fixed T (0.698 vs 0.704). Feeding
relabeled successes to the teacher's posterior inflates it — dropped; the
posterior sees requested-task evidence only. An early GSM8K claim of
teacher steering was retracted when review found the sampler epoch-frozen;
the fixed run confirmed the effect properly. Recycled-gradient exactness
is a property of the environment's relabel map: math and mazes admit exact
relabels; noisier verifiers will land between our +0.22 (fixed pools) and
+0.01 (one-shot) endpoints. LLM results are single-seed at 360M; the
2×2×2 and multi-seed replications are the active work.

## 10. Reproducibility

Every proposition has a Monte-Carlo verification script; every experiment
writes JSON/JSONL artifacts committed to the repository; the documentation
passed a two-round adversarial audit (every GPU-log-backed number
reproduced exactly; the defects found were prose qualifiers, fixed). The
framework is a dependency-light package (`frontier_rl/`, numpy-only core)
with adapters for LLM pools (verl), gym control, IsaacLab parallel sim,
and flow-policy VLAs.

---

*Repo: https://github.com/linjiw/curriculum-maxrl · Site:
https://linjiw.github.io/curriculum-maxrl/ · Base: MaxRL (arXiv:2602.02710)*
