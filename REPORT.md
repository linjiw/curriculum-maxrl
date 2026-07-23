# Curriculum-MaxRL: Experiment Report and Research Assessment

*Status: 2026-07-22. CPU numbers reproduce from this repo; historical GPU
tables retain provenance in `curriculum_maxrl/maze_gpu/` and the audit caveat below.*

**Audit note (2026-07-21):** exact binomial analysis found that the practical
drop-both estimator targets order `N-1`, not `N`; hindsight cosine hid a scale
bias; and historical GPU code counted all-pass groups as dead, used the
legacy `u_{N+1}` frontier score, trained all levels with the deepest response
budget, and scaled dense-hindsight loss with relabel count. Historical GPU AUC
was step-indexed without the post-SFT anchor despite wall-clock-matched
endpoints. CPU results remain reproducible. GPU tables
below are historical exploratory evidence pending a corrected `advmass` rerun.

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

The corrected chain now separates proof, CPU/Gym evidence, and historical GPU
evidence. The exact `u_N` teacher and corrected hindsight implementation have
not yet received a clean GPU rerun:

| Math result (PROOFS.md) | Algorithm component | CPU evidence | GPU evidence |
|---|---|---|---|
| **P1**: raw/always-CV estimators target `T=N`; practical drop-both targets `T=N-1` | explicit estimator conventions | exact binomial error ≤1.33e-15 | production estimator audit required |
| **P2**: E[Σ\|w\|] = 2(pass@N − pass@1), exact coefficient mass | AdvMass teacher utility | corrected γ=1 run beats uniform on the skill chain (V2: AUC 0.700 vs 0.650) | historical teacher used nearby `u_{N+1}`; exact run pending |
| **P2**: peak at p\* ≈ ln N/N, strictly concave | compute-indexed ZPD band | historical p̂ roughly preserved the frontier ordering; no calibrated error bound | historical diagnostic only |
| **P3**: greedy water-filling exact for fixed-p, bounded, one-step Σ mass | `allocate_rollouts_greedy` | theorem and boundary regression; older +18% pilot is archived without a retained driver | (phase 2 — needs per-prompt n in rollout worker) |
| **P4**: RLOO mass = 2p(1−p) ≡ SFL learnability | unifies curriculum literature w/ estimator algebra | exact MC match | — (interpretive) |
| **P5**: MaxRL coefficient mass / RLOO ∈[1,N−1] | finite-N comparison | exact enumeration | GRPO/MaxRL interaction remains empirical, not implied by mass alone |
| **P6**: law equality is sufficient for hindsight exactness; moment equality is necessary/sufficient | centered or success-only relabeling | centered mean cosine 1.000 but scale 0.94–1.04; success-only scale 1.000 on-chain | historical dense auxiliary MLE point estimate is confounded; corrected rerun pending |
| **P7**: floor guarantees revisit time/tail | floor 0.1 | exact visitation lemma; V3 floor flat 0–0.4 | defaults shipped in verl module |
| V6 (empirical): learning compounds ⇒ γ>1 | `power` knob, sample ∝ u^γ | γ=4: AUC 0.782 vs 0.728 | historical legacy-score GPU run was negative; corrected exact run pending |

The two places the chain *broke* were as informative as where it held:

- **Adaptive truncation T**: the historical implementation dropped its
  control variate outcome-dependently, so its population weight was
  `w_T(p)-(1-p)^(N-1)`, not `w_T(p)`. Its small negative result (0.698 vs
  0.704) is real for that implementation but does not test the claimed
  order-`T` objective; the interpretation is now inconclusive.
- **Hindsight→teacher feedback**: V4 pre-registered "safe but redundant, risk
  = optimism inflation"; the GPU run reproduced that signature exactly
  (pseudo-count p̂ 0.81 vs true 0.47 at level 2, worse final). Dropped. The
  teacher estimate must see only requested-task evidence.

Two historical CPU predictions appeared to transfer to GPU, but the audit
confounds prevent treating that as calibrated validation until the corrected
GPU factorial reproduces them.

## 3. Headline findings

### F1. Historical GPU teacher evidence is promising but mechanism-open
At matched wall-clock, every teacher variant beats uniform sampling
(AUC 0.221–0.223 vs 0.211–0.216; 6/6 paired per-seed deltas positive). The
historical mechanism diagnostic suggests fewer zero-weight groups (5.8→3.4
of 8) and ~30% more steps/sec, but that counter mixed `K=0` and `K=N`; the
corrected rerun must establish the true dead-group change. Historical sampling
mass sits on the levels where 2(pass@N − pass@1) is largest.

### F2. Historical evidence raises an objective/curriculum compatibility hypothesis
We predicted a frontier teacher would rescue GRPO from pass@k collapse by
retiring mastered prompts. In the historical logs, frontier+GRPO declined
more (pass@8 0.332→0.269) than uniform+GRPO (0.351→0.312), while MaxRL under
the teacher grew coverage (0.316→0.348). Because those runs used the audited
legacy protocol and are not a controlled causal comparison, they do not show
that the curriculum caused the difference or that likelihood-style weighting
is generally required for safety. They motivate a preregistered interaction
experiment: objective × teacher, with matched transitions, response budgets,
initialization, and corrected logging.

### F3. Hindsight beats a strong true-pass-rate priority baseline on the skill chain
The retained `frontier_rl.examples.run_skill_chain` driver reports checkpoint
means of 0.851 for a true-pass-rate proportional-priority baseline and 0.890
for discounted pseudo-count priority+γ=4+hindsight under the same trainer and
five seeds. This is not an upper bound over sampling policies. Hindsight adds
verifier-valid auxiliary targets (V1: centered relabeled gradients
are direction-aligned but scale-biased; success-only is exact on the special
skill-chain law). The categorical version: in a
frontier-heavy pool (max p = 10⁻⁵), uniform, DAPO dynamic sampling, and the
plain teacher all flatline at *exactly zero* while teacher+hindsight reaches
0.98 — there was nothing to sample toward; the signal had to be created.
Traced, hindsight invents the curriculum below the given pool, ignites
learnability within ~400 groups, then goes silent. **It is a cold-start
igniter, not a crutch.**

### F4. Proportional-to-mass sampling under-exploits (compounding)
Sampling ∝ u^γ with γ≈4 beats γ=1 (0.782 vs 0.728) and beats hard top-k.
Proportional sampling is a smooth priority baseline, not the one-step
argmax. Learning compounds —
progress on the highest-mass task unlocks successors—so sharper concentration
can help. The mass ordering and γ=4 were best among tested settings on the
chain; the right concentration remains task-graph dependent. The corrected
MountainCar factorial independently supports γ=4 over γ=1, while the
historical GPU maze ablation favored γ=1. Concentration is therefore an
empirical property of transfer structure, not part of the coefficient-mass
theorem.

### F5. Corrected MountainCar separates mechanisms—but not score families

Across ten paired seeds and at least 500k transitions, exact-mass γ=4 beats
uniform by +0.141 mean-pass AUC [95% paired bootstrap CI 0.076, 0.202] and
γ=1 by +0.116 [0.060, 0.172]. Centered and success-only hindsight add
+0.191 [0.155, 0.231] and +0.197 [0.160, 0.238], respectively. A task-agnostic
shared policy beats the disjoint per-bin control by +0.492 [0.464, 0.522].
These effects remain supported by exact sign-flip tests after Holm correction
over nine AUC contrasts.

The equally important negative is specificity: at γ=1, exact `u_N` is not
separated from uniform (+0.025 [−0.035, 0.091]), legacy `u_{N+1}`, or
learnability. The derivation identifies the correct theoretical mass, but
this experiment does not show that its small fixed-N shape difference is
empirically superior. Success-only has the best AUC point estimate (0.727 ±
0.023), yet centered versus success-only is also not separated after family
correction. This is a custom binary-threshold study on official dynamics, not
standard Gymnasium return.

### F6. Paired ablation supports both components, with diminishing returns

A new retained 12-seed skill-chain factorial gives checkpoint means of 0.660
for uniform/no hindsight, 0.732 for the proportional `u_N`-score teacher, 0.781 for
the concentrated teacher, 0.866 for uniform+centered hindsight, and 0.886 for
the reference full stack at hindsight scale 1. The direct centered-hindsight
effect under the concentrated teacher is +0.1050 [0.1012, 0.1087], while the
reference full stack remains +0.0205 [0.0190, 0.0223] above uniform+centered
hindsight. Both survive Holm correction in the declared 15-contrast family.

The interactions sharpen the causal story: teacher x hindsight is -0.0586
[-0.0779, -0.0390], and concentration x hindsight is -0.0420
[-0.0482, -0.0353]. The components are complementary in the ordinary sense
that each retains a positive matched effect, but they are **subadditive**, not
synergistic. Hindsight removes much of the dead-group problem the teacher was
designed to avoid, so diminishing returns are mechanistically expected.

Centered-hindsight checkpoint mean rises from 0.832 to 0.936 over scales
0.25, 0.5, 1, 2, 4, and 8, with diminishing increments but no reversal at the
tested boundary. Scale 8 is not an identified optimum. Because scaling also
changes the auxiliary update's effective learning rate, the result establishes
sensitivity and motivates an optimizer-matched control; it does not establish
that larger recycled-data weight is universally better.

### F7. Acrobot V3 confirms a narrow shared-policy curriculum effect

The Acrobot protocol history remains cumulative; a later revision does not
erase an earlier stop:

| Acrobot protocol | Status | Retained interpretation |
|---|---|---|
| V1 | **launch gate failed** | The selected `3e-3` pilot saturated and had no all-fail groups after warmup. No V1 confirmation was authorized. |
| V2 | **development gate failed** | The every-cell learning/headroom gate failed in the disjoint controls. The six-cell confirmation did not launch, so V2 supports no transfer or capacity-control claim. |
| V3 | **registered decision supported** | Twenty paired shared-H64, no-hindsight seeds confirm only a positive local curriculum effect. |
| V4A | **stopped: feasibility gate failed** | Integrity verification passed and the fallback selected `U*=250`, but gate 3 failed in exactly 3/9 runs; V4B was not authorized or run. |

V4A was an effect-blind, scale-zero feasibility study, not a hindsight-effect
test. The independent verifier validated the immutable lock, artifact,
schedule, accounting, and saved decision. All registered gates except gate 3
passed. Gate 3 required at least ten positive, one-to-one, nonmutating
hindsight previews in every run; the three failures had preview counts `8`,
`5`, and `6`. The projected serial runtime for the proposed 90-run factorial
was `3.452702` hours and passed its gate. Because the preview gate failed,
Stage B was not authorized and was not run; V4A supplies no evidence for or
against hindsight efficacy.

The lock, artifact, and independent-report SHA-256 hashes are respectively
`b19488783e1adba8cbac44ce8256c725a4470d8108c1192f9491ecc4882f1d8c`,
`69b827dc425014f3b568186981e9c24d95158c72653125e0ade181272def2891`,
and
`c633e09df8e056f1589e631ff4d311913e1ac5594c3647790acc4b05990fca88`.
The report's top-level `all_checks_passed=true` denotes successful verifier
integrity and recomputation, not feasibility-gate success; the decision fields
are `gates.all_pass=false` and `stage_b_factorial_authorized=false`.
The frozen lock's direct-path analyzer command has a module-import defect. No
locked V4 file was changed; from the repository root, the working equivalent
is
`/tmp/curriculum-maxrl-gym/bin/python -m frontier_rl.examples.analyze_acrobot_hindsight_v4 frontier_rl/examples/acrobot_hindsight_v4a_feasibility.json --lock frontier_rl/examples/ACROBOT_HINDSIGHT_V4A_LOCK.json --output frontier_rl/examples/acrobot_hindsight_v4a_verification.json`.
The scoped correction is retained in
`frontier_rl/examples/ACROBOT_HINDSIGHT_V4_ERRATA.md`.

On V3's primary normalized target-uniform mean-pass AUC over actual environment
transitions, including initialization, uniform scored `0.648669` and the
frontier-`u_16` coefficient-mass teacher scored `0.685021`. The paired effect
was `+0.0363524`, with a
paired-seed bootstrap 95% interval `[0.0164536, 0.0553949]` and exact two-sided
paired sign-flip `p=0.00263977` (`n=20`). The observed mean crossed the
registered `+0.03` decision threshold and the test crossed `p<=0.05`, so the
preregistered efficacy decision is supported and a positive shared-policy
effect is confirmed. Because the interval's lower endpoint is `0.0164536`, it
does **not** establish that the population effect exceeds `+0.03`. The
secondary final mean-pass averages were `0.864258` for uniform and `0.916992`
for the teacher.

The immutable artifact SHA-256 is
`30da4b9759828acb9357f5518a48196a6be98d314dffb7830b0ba4f89a31e423`; the
independent verification-report SHA-256 is
`1bb604925b36050c6b1520fce847919d7962ec8f0e300b50de49242b70b7b394`.
All verification checks passed. This result is confined to the fixed
eight-threshold [Gymnasium `Acrobot-v1`](https://gymnasium.farama.org/environments/classic_control/acrobot/)
family, one task-agnostic shared H64 policy, the frozen transition budget, and
no hindsight. It does not establish transfer causality, an advantage from
parameter sharing or capacity, hindsight efficacy or scale, wall-clock
efficiency, or generalization beyond Acrobot. The failed V1/V2 controls cannot
be rehabilitated by V3.

### F8. Historical GPU leaderboard (provisional, seed 0)

| rank | config | final | best | legacy unanchored step-AUC | pass@8 |
|---|---|---|---|---|---|
| 1 | frontier_alp + maxrl + **dense hindsight** | **0.258** | **0.269** | 0.236 | 0.361 |
| 2 | frontier_alp + maxrl | 0.244 | 0.257 | 0.233 | 0.361 |
| 3 | frontier + maxrl + hindsight | 0.230 | 0.256 | 0.234 | 0.356 |
| … | uniform + maxrl (reference) | 0.225 | 0.233 | 0.214 | — |
| … | uniform + grpo (reference) | 0.230 | 0.237 | 0.216 | 0.312↓ |

## 4. How far are we from the research goal?

**Achieved:**
1. A corrected derivation of the practical coefficient-mass utility, including
   the N−1 objective shift, N=2 learnability connection, and bounded myopic
   allocation theorem. ✅
2. Exact enumeration tests plus CPU skill-chain evidence that the teacher and
   hindsight can improve learning speed. ✅
3. A corrected Gymnasium factorial comparing shared-versus-per-bin behavior,
   task priority, and hindsight, including paired bootstrap intervals, exact
   sign-flip tests, and family-wise correction. The capacity/data-flow-mismatched
   shared-versus-per-bin arm diagnoses a transfer channel; it does not causally
   isolate transfer. Final statistics are reported in `VALIDATION.md`. ✅
4. A sealed 20-pair neural Acrobot confirmation supporting the preregistered
   shared-H64 curriculum-efficacy decision, without reviving the failed V1/V2
   transfer and capacity controls. ✅
5. A retained 12-seed component/scale ablation separating teacher,
   concentration, hindsight, estimator, and interaction effects, with all
   declared contrasts reported under one Holm family. ✅
6. Production fixes for all-pass gating, RNG/checkpoint state, verifier-backed
   relabel contracts, BFS-depth hindsight, cumulative GPU counters, contiguous
   post-filter verl teacher indices, strict feedback validation, and stateful
   mid-epoch sampling. The two verl patches passed a local application check
   against an official MaxRL checkout; the upstream commit hash was not
   retained with the result. ✅

**Not yet achieved:**
1. **The deep frontier remains uncrossed at fixed budget** — level 6+
   (distance ≥ 16) sits at ~0.01–0.02 after 2400 s in every config. Dense
   historical hindsight improved shallow levels, but the old relabeler used
   path length rather than BFS depth. Whether the remaining barrier is
   duration, exploration, or representation is unresolved.
2. **LLM-scale transfer** — the verl integration is implemented and locally unit-tested but
   data-scarce GSM8K regime is precisely where hindsight's fixed-prompt-set
   compounding should shine (the infinite-data maze understates it), but the
   single A10G cannot fit the 8-GPU recipe. Open until a larger node.
3. **Inference-efficiency currency** — historical checkpoint results exist,
   but corrected teacher/hindsight checkpoints are needed before comparison.
4. **P3 (myopic fixed-p rollout allocation) has no GPU test**—needs per-prompt
   group sizes in the rollout path.
5. **Score specificity** — exact `u_N`, legacy `u_{N+1}`, and learnability are
   empirically tied at γ=1 on MountainCar; a regime with smaller N or more
   sharply separated pass rates is needed to test the distinctive shape.

**Assessment: the mechanism is promising but not yet established at scale.**
What is proved is the estimator algebra and the myopic proxy objective. What is
supported empirically is CPU/Gym improvement in aligned shared-skill tasks.
Corrected GPU and LLM validation, stronger baselines, and more seeds remain
load-bearing.

## 5. The most promising next push (and why)

For the Acrobot track, V4A is complete and independently verified. Its
integrity checks passed, the registered fallback selected `U*=250`, and all
gates except the every-run preview-count gate passed. Exactly three of nine
runs fell below the required ten previews (`8`, `5`, and `6`), so the frozen
protocol stopped. V4B was not authorized or run. Because every V4A condition
used hindsight scale zero, this decision says only that the planned factorial
failed its feasibility gate; it is not a hindsight-effect result.

Ranked by expected information per GPU-hour:

1. **Corrected teacher factorial:** uniform vs exact `u_N` vs legacy
   `u_{N+1}` vs learnability, common floor and γ=1, no hindsight, paired SFT.
2. **Concentration ablation:** exact `u_N` at γ=1 vs γ=4 only after (1)
   establishes a teacher effect.
3. **Hindsight factorial:** none vs centered vs success-only/positive-only,
   using maximum BFS depth and cumulative K=0/K=N accounting.
4. **Optimizer-matched hindsight scale:** cross auxiliary scale with base
   learning rate or normalize update compute so scale is not merely a larger
   step-size treatment.
5. **Multi-seed replication:** expand only the effects whose paired pilot
   intervals exclude zero.
6. **LLM 2×2:** curriculum × {MaxRL, GRPO} when hardware permits.
7. **Independent Gym replication:** rerun a transition-matched survival or
   sparse-goal environment with a neural shared policy; the old CartPole
   group-matched smoke result is not evidence.

## 6. Threats to validity (kept current)

- Maze GPU results are historical seed-0 runs plus three-seed point estimates;
  no “champion” or reliability claim survives the protocol audit (mitigation:
  corrected item #4 above).
- CPU effect sizes use exact gradients and semantically exact nested labels.
  Centered scale bias shows that the full relabeled/fresh joint laws do not
  match even there; success-only unit scale holds only under the special chain
  law. Real relabelers can introduce larger bias. The V10 scale sweep also
  changes effective auxiliary learning rate and does not match optimizer
  compute; it is sensitivity evidence, not an optimizer-controlled optimum.
- Matched wall-clock on a shared machine: another process held the GPU for
  part of one window; all compared runs used exclusive windows (checked via
  dead/step + steps-per-second consistency).
- The teacher assumes a finite prompt pool with a stationary index; streaming
  pools need the parametric (ALP-GMM-style) variant sketched in GUIDE.md.
- MountainCar uses official dynamics but custom nested binary predicates, a
  weak tile policy, and a per-bin control that is not capacity/data matched.
  Its nine AUC comparisons are paired and Holm-corrected, but ten seeds still
  leave wide flag-pass uncertainty for no-hindsight conditions.
- The production sampler/index/checkpoint contracts have unit coverage and the
  patches passed a local application check, but its upstream commit hash was
  not retained and no multi-worker, multi-GPU verl training job was available
  locally for an end-to-end resume test.
