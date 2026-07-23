# Validation results and evidence audit

Companion to PROOFS.md. `run_validation.py` reproduces V0–V4;
`run_baselines.py` reproduces V5; the retained Gym/Grid/skill-chain runners and
JSON artifacts reproduce V8–V10. Sections explicitly labeled **archived
exploratory** have no retained driver/output and are excluded from the
reproducible evidence set. Skill-chain results use 5 seeds unless noted.

Metric convention also differs across generations of the work: V2–V4 use
the arithmetic mean of equally spaced post-update checkpoints (historically
called “AUC”), while V5 and V8–V9 use trapezoidal AUC with a pre-training
anchor. Comparisons are only made within a common convention.

## V0 — Exact estimator/objective audit

Exact binomial enumeration with a Bernoulli-logit score (no Monte Carlo)
confirms, to maximum error `1.33e-15`, that:

| estimator | expected objective order |
|---|---|
| raw success average `1{K>0} r/K` | `T=N` |
| Eq. 10 control variate retained at `K=0` | `T=N` |
| practical Algorithm-1 drop-both weights used here | **`T=N-1`** |

The same enumeration verifies
`E[Σ|w|]=2(pass@N-pass@1)`. This closes the earlier proof gap and is now a
regression test in `frontier_rl/test_framework.py`.

## V1 — Hindsight direction and scale (diagnoses Prop. 6)

Probe: a level-11 task at true p = 0.0005 (99% of groups dead). Relabel each
dead group to its deepest correct prefix j; compare the relabeled gradient to
the *exact* success-conditioned ML gradient of the level-j task, and to
fresh practical groups sampled directly on that task:

| prefix j | n groups | centered per-group cosine | mean cosine | centered mean scale | success-only mean scale | fresh practical mean scale |
|---|---:|---:|---:|---:|---:|---:|
| 8 | 1811 | 0.861 ± 0.071 | **1.000** | 1.043 | **1.000** | 1.001 |
| 9 | 1876 | 0.948 ± 0.016 | **1.000** | 0.960 | **1.000** | 0.530 |
| 10 | 276 | 0.956 ± 0.012 | **1.000** | 0.941 | **1.000** | 0.072 |

**Reading:** cosine alone hid a scale bias. Centered hindsight is extremely
well aligned but its scale deviates by roughly ±6% (high at prefix 8, low at
prefixes 9–10); an unconditional fresh
practical group is heavily attenuated on harder prefixes because most groups
are dead. The success-only relabeled update has unit scale on this special
skill chain, matching the exact-ML corollary in Proposition 6. This does not
establish general unbiasedness: data-dependent goal selection can change the
successful-trajectory law in other environments.

## V2 — True-pass-rate priority gap (what tracking costs)

Teacher variants, identical training loop, utility = advantage mass:

| teacher | AUC | final | advantage mass collected |
|---|---|---|---|
| uniform | 0.650 ± 0.006 | 0.966 | 720 |
| discounted pseudo-count teacher (decay 0.9) | 0.700 ± 0.005 | 0.979 | 838 |
| **true-p proportional priority** | **0.851 ± 0.002** | 0.985 | 841 |

Two lessons. (1) The Thompson teacher already collects **nearly as much
advantage mass as this true-p baseline** (838 vs 841) under the same
proportional rule and floor. (2) Yet the true-p baseline's AUC is far higher:
*when* you collect matters, not just how much. The true-p baseline moves to
each level as soon as the rule favors it; the pseudo-count estimate arrives
roughly one tracking-lag later. **The remaining gap is a
tracking problem within this priority family, not a global sampler bound.**

## V2b — Closing the tracking gap (partly archived exploratory)

- **Faster decay — archived exploratory sweep (driver not retained)**
  (pseudo-counts forget faster → track the moving student):
  AUC 0.681 / 0.700 / **0.728** / 0.719 / 0.727 at decay 0.95 / 0.9 / 0.7 /
  0.5 / 0.3. Decay 0.7 closes ~19% of the true-p priority gap; too-fast
  decay adds sampling noise back.
- **Monotone (isotonic) projection — archived exploratory result**:
  difficulty is ordered within a chain, so true pass rates are monotone in
  level. A one-off prototype projected Thompson draws onto nonincreasing
  sequences (pool-adjacent-violators) and reported AUC 0.711 at decay 0.9
  and 0.733 combined with decay 0.7. The prototype script was not retained,
  so these numbers are **not part of the reproducible evidence in this
  repository**. Reimplementing and preregistering this ablation is follow-up
  work; it remains a motivated way to share strength across ordered tasks.

## V3 — Explore/exploit floor curve (validates Prop. 7's design reading)

AUC is *flat* (0.699–0.712) for floor ∈ [0, 0.4] and only collapses at
floor = 1 (pure uniform, 0.638). The coefficient-mass utility with randomized
pseudo-count draws is self-exploring — uncertainty already probes
uncertain arms, so the floor adds little *on this testbed* (no distribution
shift, no forgetting pressure: mastered skills stay mastered because
gradients never reverse them). Keep the floor for real settings — Prop. 7's
staleness bound is about *re-detecting* change, which this toy cannot
exhibit — but expect low sensitivity to its exact value.

## V4 — Hindsight→teacher feedback (CPU analog of `--hindsight-to-teacher`)

| variant | AUC | final | dead-group rate |
|---|---|---|---|
| hindsight only | 0.874 ± 0.004 | 0.984 | 0.041 |
| + feedback to pseudo-counts | 0.864 ± 0.003 | 0.983 | 0.041 |

**Slightly negative on the toy** — and V2 explains why: the tracking lag it
was designed to fix is invisible here because hindsight already trains the
prefix tasks, whose pseudo-counts then update from *natural* successes almost
immediately. Dead-group rate does not rise (no runaway optimism), so the
mechanism is safe but redundant on this testbed. Prediction for the GPU
A/B/C: feedback matters only if natural successes lag relabeled competence
(long-horizon regimes); treat C ≤ B on the maze as consistent with this.

## V5 — Head-to-head vs DAPO dynamic sampling + regime map (`run_baselines.py`)

Matched *generation* budget (3200 groups, discarded draws count), 5 seeds.
AUC is a transition-indexed trapezoid including the pre-training baseline.
DAPO-style dynamic sampling = redraw prompts until the group is live
(0 < K < N), paying for every draw. Three task-pool regimes:

| regime | uniform+maxrl | dapo+maxrl | teacher+maxrl | **teacher+maxrl+hindsight** |
|---|---|---|---|---|
| easy-heavy (levels 1–6), AUC | 0.880 | 0.880 | 0.887 | **0.912** |
| balanced (1–12), AUC | 0.645 | 0.645 | 0.699 | **0.863** |
| frontier-heavy (5–12), AUC | 0.000 | 0.000 | 0.000 | **0.860** |
| frontier-heavy, final | 0.000 | 0.000 | 0.000 | **0.981** |

**Findings:**

1. **The full method is best in every tested regime**, and its margin grows
   with difficulty: +0.025 AUC easy-heavy, +0.164 balanced, and
   **0.860 vs 0.000 frontier-heavy**.
2. **The frontier-heavy result is categorical, not incremental.** With max
   initial pool pass rate 10⁻⁵, the expected number of live groups in the
   whole budget is ≈0.5 — uniform, DAPO, and the plain teacher all flatline
   at exactly 0 because there is *nothing to sample toward*: DAPO's redraws
   and the teacher's predictions both only reallocate compute among tasks
   none of which can produce a success. Only hindsight *creates* signal.
3. **The bootstrapping mechanism, traced:** in the first ~140 groups
   hindsight relabels dead groups to prefix tasks (61 below the pool — i.e.
   it *invents the missing curriculum* below the given distribution — and
   81 in-pool); those prefix gradients raise in-pool pass rates into
   learnability; by draw 400 there are already 257 live groups and normal
   MaxRL takes over; hindsight then goes almost silent (81→81 relabels from
   draw 800 to 3200). It is a cold-start igniter, not a permanent crutch.
4. **DAPO comparison correction:** after fixing evaluation timestamps, DAPO
   is exactly equal to uniform here at a matched total generated-group budget.
   That equivalence is expected in this sequential simulator: both consume the
   same uniform stream, discard the same degenerate groups, and update on the
   same live groups; DAPO's redraw-until-live batching changes no update or
   charged sample. The previously reported DAPO AUC advantage was an
   irregular-checkpoint artifact.

## V6 — Utility concentration (core contrast retained; extended sweep archived)

`frontier_rl/examples/run_skill_chain.py` reproduces γ=1 (0.728) and γ=4
(0.782) inside the reusable trainer. The γ=2, γ=8, and hard top-k columns came
from an older unretained sweep and remain archived exploratory values:

| γ (power) | 1 | 2 | 4 | 8 | top-k hard selection |
|---|---|---|---|---|---|
| AUC | 0.728 | 0.764 | **0.782** | 0.782 | 0.771 |

The retained contrast shows γ=4 beating proportional γ=1 on this chain. The
archived columns suggested saturation and a small hard-top-k reversal, but
those details require a retained rerun. **Why the temperatures can differ:** `u(p)` orders expected coefficient
mass per group, but proportional sampling is a design choice—not its
one-step maximizer. *Learning* compounds: gradient steps on the single
highest-mass task advance the frontier, which unlocks the next task, and
the compounding favors concentrating beyond linear in this testbed. The mass
functional is a useful ordering here; the retained best temperature is sharper
than proportional. γ remains an empirical task-graph knob, not a general
default theorem.

**V6b — archived exploratory ODE (driver not retained).** Abstract the chain to
skill parameters θ₁..θ_L with p_j = Π_{i≤j} σ(θᵢ) and shared-skill
gradients (training level j updates all θ_{i≤j} ∝ q_j·u(p_j)). This
two-line model reproduces the γ effect quantitatively: chain AUC
0.320→0.415→0.426 at γ=1→4→8 (saturation at γ≈4–8, exactly as the full
testbed) — while on a *flat* pool (no shared skills, heterogeneous
difficulty) the effect nearly vanishes (0.409→0.426→0.427 with *worse*
finals at high γ). This supports the hypothesis that shared structure drives
γ>1's advantage rather than Thompson noise or the estimator. It is supporting
evidence, not a general
causal theorem; treat γ as an empirical task-graph knob.

## V7 — Full stack vs true-p proportional priority

All rows below reproduce in one retained driver:
`python3 -m frontier_rl.examples.run_skill_chain`. Its historical “AUC”
convention is the mean of equally spaced post-update checkpoints, as noted at
the top of this document.

| config | AUC (5 seeds) |
|---|---|
| Thompson teacher, γ=1 | 0.728 |
| Thompson teacher, γ=4 | 0.782 |
| true-p proportional priority, γ=1 | 0.851 |
| Thompson γ=1 + hindsight | 0.880 |
| **Thompson γ=4 + hindsight** | **0.890 ± 0.002** |

**The full stack beats this strong true-p baseline.** It is not an upper bound
over adaptive sampling policies. Hindsight adds verified auxiliary targets
from failures, changing what a sampled group can yield.
Concentration remains beneficial with hindsight (+0.010 in this five-seed
checkpoint convention), but the paired factorial in V10 shows that the two
effects are subadditive rather than synergistic.
This is the strongest statement the CPU testbed can make: **the proposed
method's largest toy advantage is not just better allocation of the standard
signal; it is access to an auxiliary relabeling signal.**

## V8 — Corrected Gymnasium MountainCar validation

Protocol: official `MountainCar-v0` dynamics with ten custom nested binary
thresholds, one task-agnostic shared tile policy, 500,000 actual environment
transitions per run (the final complete group can overshoot slightly), ten
paired seeds, 64 evaluation episodes per threshold, fixed side-effect-free
evaluation seeds, `N=16`, and a nine-condition factorial. AUC is normalized
trapezoidal area over actual transition checkpoints. Values are mean ± sample
SD; brackets are 95% percentile-bootstrap CIs for the mean. Raw curves, group
counts, paired deltas, and provenance are in
`frontier_rl/examples/mountaincar_shared_validation.json`.

| config | mean-pass AUC | final mean pass | final flag pass |
|---|---:|---:|---:|
| flag-only shared | 0.024 ± 0.006 [0.020, 0.028] | 0.024 ± 0.006 | 0.000 ± 0.000 |
| uniform curriculum shared | 0.389 ± 0.071 [0.347, 0.431] | 0.684 ± 0.094 | 0.058 ± 0.079 |
| exact mass, γ=1, shared | 0.414 ± 0.081 [0.367, 0.462] | 0.758 ± 0.127 | 0.208 ± 0.266 |
| legacy `u_{N+1}`, γ=1, shared | 0.414 ± 0.078 [0.371, 0.464] | 0.745 ± 0.121 | 0.175 ± 0.274 |
| learnability, γ=1, shared | 0.411 ± 0.037 [0.389, 0.432] | 0.697 ± 0.088 | 0.080 ± 0.143 |
| exact mass, γ=4, shared | 0.530 ± 0.059 [0.493, 0.561] | 0.928 ± 0.056 | 0.664 ± 0.232 |
| exact γ=4 + centered hindsight, shared | 0.720 ± 0.029 [0.704, 0.737] | 0.969 ± 0.013 | 0.842 ± 0.062 |
| **exact γ=4 + success-only hindsight, shared** | **0.727 ± 0.023 [0.713, 0.740]** | **0.970 ± 0.014** | **0.848 ± 0.058** |
| exact γ=4 + centered hindsight, per-bin params | 0.229 ± 0.031 [0.209, 0.245] | 0.284 ± 0.028 | 0.000 ± 0.000 |

The paired AUC effects in the reported nine-contrast family are:

| contrast | mean delta [95% bootstrap CI] | exact sign-flip `p` | Holm-adjusted `p` |
|---|---:|---:|---:|
| exact γ=1 − uniform | +0.025 [−0.035, +0.091] | 0.488 | 1.000 |
| exact γ=4 − uniform | **+0.141 [+0.076, +0.202]** | 0.0078 | **0.0469** |
| exact γ=4 − exact γ=1 | **+0.116 [+0.060, +0.172]** | 0.0078 | **0.0469** |
| exact γ=1 − legacy γ=1 | −0.000 [−0.023, +0.026] | 0.988 | 1.000 |
| exact γ=1 − learnability γ=1 | +0.003 [−0.054, +0.063] | 0.928 | 1.000 |
| centered hindsight − no hindsight | **+0.191 [+0.155, +0.231]** | 0.0020 | **0.0176** |
| success-only hindsight − no hindsight | **+0.197 [+0.160, +0.238]** | 0.0020 | **0.0176** |
| centered − success-only hindsight | −0.006 [−0.021, +0.008] | 0.4355 | 1.000 |
| shared − per-bin centered | **+0.492 [+0.464, +0.522]** | 0.0020 | **0.0176** |

After family-wise correction, the supported claims are narrower and more
useful: `γ=4` beats both uniform and `γ=1`; each hindsight variant beats no
hindsight; and shared parameters beat the disjoint-table control in this
implementation and budget. Proportional (`γ=1`) exact mass is not separated
from uniform, legacy `u_{N+1}`, or learnability. Success-only has the larger
point-estimate AUC, but centered versus success-only is not separated by the
exact family-corrected test. The per-bin control has more parameters and
different data flow, so it diagnoses a transfer channel here rather than
proving that task identity is universally harmful. These are custom binary
threshold metrics on official dynamics, not standard Gymnasium return.

## V9 — Corrected goal-conditioned gridworld replication

Ten seeds, 150 matched group steps (not transition-matched), fixed
non-mutating evaluation, one concrete
achieved goal per relabeled group, and a frozen-policy batch gradient:

| config | mean-pass AUC | final mean pass |
|---|---:|---:|
| uniform + practical MaxRL | 0.583 ± 0.022 | 0.857 ± 0.021 |
| `u_N`-score teacher + practical MaxRL | 0.652 ± 0.020 | 0.912 ± 0.024 |
| **teacher + centered hindsight** | **0.702 ± 0.032** | **0.916 ± 0.038** |

Positive relabeled trajectories are cut at their first hit of the rewritten
goal, matching the stopping rule of a fresh reach-task rollout. The result
supports transfer on a genuinely goal-conditioned policy, but the
tabular grid remains a mechanism check rather than a robotics
benchmark. Verifier-valid labels and goal rewriting hold; full joint-law
equality is not claimed. Raw per-seed curves are in
`frontier_rl/examples/grid_reach_validation.json`.

## V10 — Paired component and hindsight-scale ablation

`python3 -m frontier_rl.examples.run_skill_chain_ablation` runs a retained
12-seed factorial on the shared-skill chain. Every arm receives 400 trainer
steps, 3,200 requested groups, and 51,200 rollout attempts. Evaluation is the
analytic mean pass rate at steps 0, 10, ..., 400 and cannot perturb training
randomness. The primary metric below is the arithmetic **checkpoint mean**
including step zero; it is not called AUC.

| condition | checkpoint mean [95% seed-bootstrap CI] | final mean pass |
|---|---:|---:|
| uniform, no hindsight | 0.660 [0.645, 0.678] | 0.967 |
| `u_N`-score teacher, gamma=1, no hindsight | 0.732 [0.724, 0.741] | 0.980 |
| `u_N`-score teacher, gamma=4, no hindsight | 0.781 [0.777, 0.785] | 0.983 |
| uniform + centered hindsight | 0.866 [0.864, 0.867] | 0.979 |
| `u_N`-score teacher, gamma=1 + centered hindsight | 0.879 [0.877, 0.880] | 0.984 |
| **reference full stack: gamma=4 + centered hindsight, scale=1** | **0.886 [0.885, 0.887]** | **0.986** |
| gamma=4 + success-only hindsight, scale=1 | 0.885 [0.883, 0.886] | 0.986 |

The pre-specified paired checkpoint-mean effects include:

| contrast | mean delta [95% paired-bootstrap CI] | Holm-adjusted p |
|---|---:|---:|
| gamma=1 teacher - uniform, no hindsight | +0.0718 [0.0531, 0.0901] | 0.0073 |
| gamma=4 - gamma=1, no hindsight | +0.0494 [0.0422, 0.0563] | 0.0073 |
| centered hindsight - none under gamma=4 | +0.1050 [0.1012, 0.1087] | 0.0073 |
| reference full stack - uniform + centered hindsight | +0.0205 [0.0190, 0.0223] | 0.0073 |
| teacher x centered-hindsight interaction at gamma=1 | -0.0586 [-0.0779, -0.0390] | 0.0073 |
| gamma x centered-hindsight interaction | -0.0420 [-0.0482, -0.0353] | 0.0073 |
| success-only - centered hindsight under gamma=4 | -0.00145 [-0.00252, -0.00057] | 0.0083 |

All 15 declared contrasts, including the five scale contrasts below, survive
Holm correction for one family of exact two-sided paired sign-flip tests. The
scientific reading is not "synergy": the teacher, concentration, and hindsight
each add value in the matched local conditions, but their negative interactions
show **diminishing returns**. Centered hindsight is slightly ahead of
success-only on this chain, whereas V8 does not separate them on MountainCar;
there is no domain-general estimator ranking.

The centered-hindsight scale sweep is deliberately reported as sensitivity:

| scale | 0.25 | 0.5 | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|---:|---:|
| checkpoint mean | 0.832 | 0.858 | 0.886 | 0.908 | 0.924 | **0.936** |

Performance is still increasing at the tested boundary, although the absolute
increments diminish. Therefore scale 8 is **not** identified as optimal, and
scale 1 is not a universally validated default. Scaling this auxiliary update
also changes its effective learning rate. Rollout groups and attempts are
matched, but hindsight can add a reused-data policy update and sampled task
depth changes the number of primitive skill decisions. A follow-up should
match optimizer compute and jointly sweep the base learning rate before
attributing the curve solely to better recycled data.

The complete curves, counts, paired contrasts, protocol, and source hashes are
retained in `frontier_rl/examples/skill_chain_component_ablation.json`.

## Neural Acrobot confirmation ledger (protocol V1 through V4A)

These Acrobot protocol labels are separate from the V0--V10 validation-study
labels above. The chronology is retained so the confirmed V3 result cannot
hide the development failures that narrowed its claim:

| Acrobot protocol | Decision status | Evidence boundary |
|---|---|---|
| V1 | **stopped: launch gate failed** | The selected `3e-3` pilot saturated and had no all-fail groups after warmup. No V1 confirmatory run was authorized. |
| V2 | **stopped: development gate failed** | Five of six effect-blind gates passed, but the every-cell learning/headroom gate failed in the disjoint controls. No six-cell confirmation, transfer inference, or capacity-control claim was authorized. |
| V3 | **confirmed: registered efficacy decision supported** | Exactly 20 paired seeds tested only teacher versus uniform with one shared H64 policy and hindsight off. |
| V4A | **stopped: feasibility gate failed** | Integrity verification passed and the fallback selected `U*=250`, but gate 3 failed in exactly 3/9 runs; V4B was not authorized or run. |

V3 used the fixed eight-threshold [Gymnasium
`Acrobot-v1`](https://gymnasium.farama.org/environments/classic_control/acrobot/)
family, a nominal two-million-transition budget per run with every final group
completed, and normalized target-uniform mean-pass AUC over actual environment
transitions including update zero. The independently recomputed summaries are:

| V3 arm | pairs | primary transition-AUC | secondary final mean pass |
|---|---:|---:|---:|
| uniform, shared H64, no hindsight | 20 | `0.648669` | `0.864258` |
| frontier-`u_16` coefficient-mass teacher, shared H64, no hindsight | 20 | `0.685021` | `0.916992` |

The paired teacher-minus-uniform AUC effect is `+0.0363524`, with descriptive
paired-seed bootstrap 95% interval `[0.0164536, 0.0553949]` and exact
two-sided paired sign-flip `p=0.00263977`. The observed mean met the registered
`>=+0.03` threshold and the sole registered test met `p<=0.05`; therefore the
preregistered decision is supported and positive shared-policy curriculum
efficacy is confirmed in this local design. The interval's lower endpoint is
below `+0.03`, so it does not establish a population effect above that
threshold. Final mean pass is secondary and cannot rescue the primary rule.

The confirmatory artifact
`frontier_rl/examples/acrobot_neural_v3_shared_confirmatory.json` has SHA-256
`30da4b9759828acb9357f5518a48196a6be98d314dffb7830b0ba4f89a31e423`.
The independent report
`frontier_rl/examples/acrobot_neural_v3_verification.json` has SHA-256
`1bb604925b36050c6b1520fce847919d7962ec8f0e300b50de49242b70b7b394` and
records all checks passing.

V3 does not evaluate or establish transfer causality, a parameter-sharing or
capacity advantage, hindsight efficacy or scale, wall-clock efficiency, or
generalization beyond Acrobot. The failed V1/V2 controls remain failures and
cannot be rehabilitated by the narrower positive V3 result.

V4A was a scale-zero, effect-blind feasibility study for an optimizer-matched
hindsight factorial. The independent verifier validated the immutable source
and runtime lock, artifact, registered schedule, accounting, and saved runner
decision. The registered fallback selected `U*=250` optimizer updates. All
gates except gate 3 passed. Gate 3 required at least ten positive, finite,
one-to-one, nonmutating hindsight previews in every run; exactly three of the
nine runs had only `8`, `5`, and `6` previews. The proposed 90-run factorial's
serial runtime projection was `3.452702` hours and passed its gate.

The failed preview gate stopped the protocol. V4B was not authorized and was
not run. Because hindsight scale was zero in every V4A condition, this outcome
is neither evidence for nor evidence against hindsight efficacy. In the
independent report, top-level `all_checks_passed=true` means the verifier's
integrity and recomputation checks passed; the feasibility decision is
`gates.all_pass=false` and `stage_b_factorial_authorized=false`.

V4A SHA-256 provenance is:

- lock: `b19488783e1adba8cbac44ce8256c725a4470d8108c1192f9491ecc4882f1d8c`
- artifact: `69b827dc425014f3b568186981e9c24d95158c72653125e0ade181272def2891`
- independent report: `c633e09df8e056f1589e631ff4d311913e1ac5594c3647790acc4b05990fca88`

The direct file-path analyzer command in the frozen lock has a module-import
defect. No locked V4 file was changed. From the repository root, the exact
working invocation is:

```bash
/tmp/curriculum-maxrl-gym/bin/python -m frontier_rl.examples.analyze_acrobot_hindsight_v4 \
  frontier_rl/examples/acrobot_hindsight_v4a_feasibility.json \
  --lock frontier_rl/examples/ACROBOT_HINDSIGHT_V4A_LOCK.json \
  --output frontier_rl/examples/acrobot_hindsight_v4a_verification.json
```

See `frontier_rl/examples/ACROBOT_HINDSIGHT_V4_ERRATA.md` for the scoped
invocation correction.

## Consolidated design updates

1. **Pseudo-count decay 0.9 → 0.7** in the reusable `frontier_rl` and verl
   `FrontierTeacher` defaults (V2b: +0.028 AUC, ~19% of the true-p gap).
   The earlier validation studies labeled V2/V3 retain decay 0.9 so their reported comparison
   remains reproducible.
2. **Isotonic projection is a follow-up**, not a currently packaged feature.
   It is applicable only when a trustworthy difficulty ordering exists
   (maze levels: yes; arbitrary GSM8K prompts: no ordering — skip).
3. **Floor default stays 0.1** (V3: harmless here, load-bearing under
   drift/forgetting per Prop. 7).
4. **Dense hindsight remains the best-supported empirical direction** (V1:
   centered gradients are aligned but scale-biased; success-only is exact on
   the special skill-chain law); the teacher-feedback loop is optional and should
   be judged by the GPU A/B/C, not assumed.
5. **Hindsight scale remains a tuning dimension, not a settled default.** V10
   improves monotonically through scale 8 on the chain, where scaling acts in
   part like an auxiliary learning-rate increase. Keep scale 1 as the reference
   configuration for comparisons, and require an optimizer-matched sweep before
   changing the production default.
