# Opus 5 Review — Curriculum-MaxRL

*Independent adversarial review, 2026-07-27. Seven reviewers (theory, implementation,
experimental design, LLM rung, claim traceability, paper/positioning, goal/critical-path),
56 raw findings, the 10 highest-severity findings put through adversarial refutation
(9 survived, 1 refuted), plus my own re-derivations and re-runs. Every number below that
says "I measured" was recomputed from the repo's own artifacts or by re-running the repo's
own harness.*

---

## 0. The research goal, as I understand it

Build on MaxRL (arXiv:2602.02710) — the success-conditioned estimator whose weight function
`w(p) = (1−(1−p)^T)/p` upweights hard prompts, an *implicit gradient-level curriculum* — and
establish three things:

1. **Derived curriculum.** The optimal *explicit* curriculum falls out of the estimator's own
   algebra: `E[Σ|w|] = 2·(pass@N − pass@1)`, a compute-indexed ZPD functional peaking at
   `p* ≈ ln N / N`. No difficulty band, threshold, or bucket hyperparameters.
2. **Signal creation.** Allocation cannot rescue all-fail groups, so add hindsight recycling:
   a failed trajectory is a verified success for the goal it actually reached.
3. **Objective safety.** The identical curriculum that helps MaxRL *hurts* GRPO — curricula
   amplify objective-level pathologies.

Target: a publishable paper plus a validated framework, on one A10G, via an escalating ladder
(CPU exact-gradient chains → 1.26M maze → Gymnasium → LLM RLVR).

**My verdict in one line: there is a real paper here, and it is not the paper that is
written. The strongest claim is being buried; the two loudest claims do not survive
ablations that take ten CPU-minutes to run.**

---

## 1. What is genuinely good (not padding — this is load-bearing)

- **The algebra is correct.** I re-derived Prop 1, 2, 4, 5 independently. `E[Σ|w|] =
  2((1−(1−p)^N) − p)` holds; `p* = 1 − N^(−1/(N−1))` gives 0.1688 / 0.1058 / 0.0375 at
  N=16/32/128, matching the docs; `u_2(p) = p(1−p)` exactly, so the "learnability is the N=2
  slice" unification is real. The MaxRL/RLOO mass ratio → N−1 as p→0 checks out numerically
  (I measured 14.99 at N=16, p=1e-4). The T=N−1 correction is right, and catching it at all
  was good work.
- **Prop 5 is stronger than claimed.** The proof's final step reduces to `(1−p)^N ≤ (1−p)²`,
  true for *all* p ∈ (0,1) when N≥2 — not just on (0, p*]. The paper understates its own result.
- **Artifact discipline is above the median submission.** Every GPU-log-backed number I
  checked reproduced from raw JSONL: maze champion 0.252±0.005 final / 0.229±0.009 AUC
  (I got 0.2516±0.0064 / 0.2288±0.0104), the V7 oracle table exactly, V5 frontier-heavy
  0.981/0.928 exactly, the CPU hindsight table exactly (0.6554/0.8778/0.7066/0.8851).
- **Real pre-registration.** Commit 39520fa is timestamped before any cell finished. That is
  rarer than it should be.
- **The P-G4 retraction (c06b9c3) is exemplary.** Finding your own sampler was epoch-frozen,
  retracting the claim it supported, and documenting it is genuine scientific hygiene.
- **The Countdown self-review found real invalidators** (dict-ground_truth crash, an
  `eval()` DoS via `**` passing a character-class regex, 28% train/test overlap) and the
  fixes landed — the AST whitelist at `countdown_reward.py:41-57` is correct.
- **Honest negatives are documented, not buried:** adaptive-T, γ non-transfer, the
  hindsight→teacher feedback loop, the refuted duration hypothesis.

---

## 2. Scorecard against the three thesis claims

| Claim | Status | Best evidence tier |
|---|---|---|
| Closed-form mass; p* ≈ ln N/N; p(1−p) is the N=2 slice | **ESTABLISHED** | Analytic + independent recomputation |
| Sampling ∝ u_N is *optimal* allocation | **CONTRADICTED** | Analytic + 10-seed paired CPU |
| Derived u_N beats published curriculum utilities | **SUGGESTIVE** | CPU only; maze doesn't test it |
| Full stack beats the oracle allocator (0.890 vs 0.851) | **CONTRADICTED** | 10-seed paired, own harness |
| Frontier-heavy: only creation works (0.00 → 0.98) | **CONTRADICTED** (mechanism) | 5-seed missing-control ablation |
| Prop 6 hindsight exactness (cosine 0.956 vs 0.958) | **UNSUPPORTED** | Probe has no discriminating power |
| H6: curriculum grows pass@k under MaxRL, decays under GRPO | **ESTABLISHED** | 9 maze runs, permutation p=0.0079 |
| LLM 2×2 confirms the safety half (.096→.093) | **UNSUPPORTED** | Inside its own noise floor |
| Teacher verifiably steering at LLM scale (0.48 vs 0.65) | **CONTRADICTED** | Min-vs-mean artifact |
| MountainCar 0.889 → 0.944 → 1.000 every seed | **UNSUPPORTED** | No artifact exists |
| Up to 11× fewer inference samples | **SUGGESTIVE** | 1 seed, 2 of 4 levels reverse |
| 65–75% of groups produce zero signal | **SUGGESTIVE** | Directionally right, counter conflates K=0/K=N |
| Countdown is a viable flagship | **CONTRADICTED** | Closed-form dose + pool triviality |

---

## 3. Blocking findings

### B1. `max_groups_per_step=8` means the *currently running* Countdown cell cannot show a hindsight effect

**This is the only urgent item.** PID 93742 is running cell C1 right now
(`cd_maxrl_curtrue_hstrue_16r`) with `+data.hindsight.max_groups_per_step=8`.

Arithmetic, from the project's own frozen probe:

| quantity | value |
|---|---|
| dead groups per 64-prompt step (dead frac ≈0.72) | ~46 |
| examined at cap=8 | 8 → **17%** |
| per-trace relabel yield (pre-registered, disclosed) | 0.16–0.69% |
| ⇒ P(examined group yields ≥1 relabel) | 0.025–0.105 |
| ⇒ **relabeled groups per step** | **0.20–0.84** |
| ⇒ share of gradient events | **1.1–4.5%** |
| ceiling even at 100% yield | 31% |
| CPU rung, where hindsight worked | ~100% of dead groups |

~43 relabel events across a 60-step cell. The pre-registration honorably flags this risk
(P-C4) and then the cell was launched anyway. **The null outcome is derivable in closed form
before spending the GPU-hours.** Fix: `max_groups_per_step ≥ 48`, one config edit.

### B2. The Countdown pool is not Countdown

`prep_countdown.py:41-52` builds every target by folding ops **left-to-right over the numbers
in the order shown**, with no parentheses and no division. Search space per tier is therefore
`3^(n-1)`:

| tier | op-sequences | random-guesser pass@16 |
|---|---|---|
| 2 operands | 3 | 0.998 |
| 3 operands | 9 | 0.848 |
| 4 operands | 27 | 0.453 |

Real 4-number Countdown is 24 perms × 5 tree shapes × 4³ ops ≈ 7,680 — a **284× reduction**.
Tier 0 has exactly 3 achievable integers. No document discloses the in-order/no-division
restriction ("all solvable by construction"), while `NEXT_EXPERIMENTS.md:13,30-31` imports its
difficulty justification from the strictly harder `Jiayi-Pan/Countdown-Tasks-3to4`.

Caveat that the verifier established and I accept: the frozen probe returns pass@16 = 0.0 on
all tiers because the binding constraint at 360M is *format emission*, not search. So the
guesser bound is an unrealized ceiling that becomes operative right after the SFT
warmstart — which itself teaches left-to-right `+,-,*` from the same generator.

Also: the relabeler never rewrites the response, so a reward-1 training example contains a
`<think>` block naming the **old** target — violating Prop 6's own conditioning contract and
creating an obvious degenerate incentive. Gate on `str(old_target) not in response`.

### B3. "Creation breaks the oracle allocation ceiling" is false as an ordering

`EVIDENCE.md:31` — "no pure sampler can exceed the oracle" — is empirically false on the
project's own harness. The arm labeled "oracle" is not a supremum over samplers; it is one
soft-proportional sampler additionally handicapped by a **10% uniform exploration floor**,
a device whose documented purpose (posterior-staleness insurance) is vacuous under perfect
information.

From `v7_oracle_result.json` itself, the honest comparison is already much narrower than the
abstract's:

| comparison | margin | paired |
|---|---|---|
| full stack 0.8895 vs oracle **γ=1** 0.8511 (abstract's number) | +0.0384 | — |
| full stack 0.8895 vs oracle **γ=4** 0.8836 (same-γ) | **+0.0059** | t=5.07, 5/5 |

And I ran the missing arm — give the oracle the same second channel:

```
oracle γ4              AUC 0.8926 ± 0.0009
oracle γ4 + hindsight  AUC 0.9016 ± 0.0019   <-- ahead of the full stack
thompson γ4 + hindsight AUC 0.8847 ± 0.0042  (the paper's "beats the oracle" arm)
```

The reviewers' 10-seed replication found five hindsight-free-or-matched arms above the full
stack (tilted oracle 0.8927 with **no hindsight**, 10/10 paired seeds, t=4.77). Worse, the
whole 0.650→0.890 ladder fits inside one arm's learning-rate envelope — I measured
`oracle γ4` at lr×2 reaching **0.9405**, and `thompson γ4 lr×2` at 0.8264.

**What survives, and it is a real result:** hindsight adds +0.003 to +0.009 *on top of the
best sampler including an oracle*. That is "allocation and creation substitute more than they
compose," which is more interesting than a ceiling story — but it is +0.006, not +0.039.

### B4. The frontier-heavy 0.00 → 0.98 result is task-pool access, not signal creation

This is the most-quoted claim in the program. I ran the control the design omits.

`run_baselines.py` restricts the *sampling* pool to levels 5–12, but hindsight relabels to
prefix tasks at levels 1–4 — tasks every baseline is **forbidden to sample**. Give the
baselines that same access:

```
V5 as published (levels 5-12 pool):
  uniform+maxrl 0.000   dapo+maxrl 0.000   teacher+maxrl 0.000   teacher+hs 0.929

MY CONTROL — full-pool sampling, frontier-pool evaluation, same budget:
  uniform+maxrl  final 0.955   AUC 0.662
  teacher+maxrl  final 0.970   AUC 0.696
```

The baselines go from *exactly zero* to 0.955 on the same evaluation set purely by being
allowed to touch the sub-pool hindsight silently trains on. Reviewers additionally measured
that **uniform+hindsight (0.9313) ≥ teacher+hindsight (0.9282)** — the teacher contributes
nothing — and that restricting relabel targets to the declared pool collapses the arm to
0.0727.

The honest claim survives and is still interesting: *recycling discovers the sub-pool a
sampler would otherwise have to be handed.*

### B5. Prop 6's validation probe cannot fail

The skill-chain policy is indexed by **skill alone** (`testbed.py:42` — `theta[s]`; the task
never enters the policy), and success is the single sequence `(actions==0).all()`
(`testbed.py:78`). So the conditional-on-success law is a point mass, identical for the
original and relabeled task: Prop 6's hypothesis holds *by construction*.

Reviewers ran the null models. Pure imitation `r/K` (knowingly biased, no baseline),
REINFORCE `r/N`, and imitating one randomly chosen success all return **byte-identical
cosines to MaxRL** — 0.885279 mean, 0.727778 min, to six decimals — because they are mutually
collinear here. Cosine 1.000 is forced arithmetic, not evidence. The headline j=10 row also
retains only 21 of 276 fresh reference groups (92.4% died and were dropped at
`run_validation.py:114`); pooled figures are 0.908 vs 0.888.

Stop citing 0.956/0.958 as evidence for Prop 6 (PAPER.md:113/:201, main.tex:114/:228-230,
EVIDENCE.md:42/:138, REPORT.md:36).

### B6. MountainCar §7.4's numbers have no artifact — this is an integrity finding

`PAPER.md:265` / `main.tex` §7.4 / `SCHEDULE.md:41`: "uniform 0.889 → teacher 0.944 → full
stack **1.000 in every seed**."

I searched every ref in the repo. `0.944` appears **only** in PAPER.md and SCHEDULE.md prose —
never in a JSON, never in a script. The only persisted artifact on main,
`frontier_rl/examples/mountaincar_scaled.json`, reports:

```
uniform+maxrl        auc 0.267  final 0.300  flag_bin 0.0
teacher(g1)+maxrl    auc 0.272  final 0.319  flag_bin 0.0
teacher(g4)+maxrl+hs auc 0.303  final 0.355  flag_bin 0.0     <-- all five arms flag_bin 0.0
```

A reviewer who greps for this finds `0.000` for every arm and rejects on that alone, then
distrusts the theory they would otherwise have accepted. Reviewers located a *better*
result on an unmerged branch (flag 0.000 → 0.058 → 0.664 → 0.842, 10 seeds,
transition-matched, Holm-corrected, SHA-256 provenance). **Delete the 1.000 today; report the
branch result verbatim.** It is more credible *and* more citable than an unsourced 1.000.

### B7. The abstract's LLM coverage claim is contradicted by the project's own newest artifact

Abstract: "grows pass@k coverage where GRPO's collapses." The uncommitted
`maxrl/curriculum_maxrl/gsm8k_partial/ksweep_results.json`:

| cell | pass@1 | pass@4 | pass@8 | pass@16 |
|---|---|---|---|---|
| grpo uniform | .0383 | .1107 | .1681 | **.2383** ← best of all four |
| grpo + teacher | .0305 | .0910 | .1434 | .2148 |
| maxrl uniform | .0327 | .0986 | .1567 | .2305 |
| maxrl + teacher | .0281 | .0844 | .1384 | .2148 |

The teacher **lowered** coverage at every k under **both** objectives. GRPO-uniform has the
best pass@16 in the grid. The claim as written is false at LLM scale.

What survives — and it is the better statistic — is the *difference-in-differences*, which is
immune to the eval-noise floor because both arms share a baseline draw:
**−0.0032 / −0.0055 / −0.0064 / −0.0078 at k = 1/4/8/16.** Consistently signed across four
nested k values. Report that.

---

## 4. Major findings

### M1. The GSM8K headline is 0.2–0.7 σ, and the noise floor is measurable from the repo's own logs

Reviewers found **five evaluations of the identical unmodified SmolLM2-360M** (four cells'
step-0 evals plus the killed cell-1), all with byte-identical eval config:

```
mean@4 = [.081, .091, .072, .078, .066]  -> SD 0.0094
pass@4 = [.166, .182, .151, .162, .136]  -> SD 0.0172
```

The headline P-G2 statistic (.096→.093 mean@4, .193→.181 pass@4) is **z = −0.32 and −0.70**.
My independent binomial estimate agrees: SE ≈ 0.009–0.019 on a 256-row × n=4 slice.

Worse: the logs contain a **zero-model-change replicate** — the same step-25 checkpoint
evaluated twice (.094/.189 in-run vs .0967/.2066 reloaded) — whose pass@4 movement (+0.018)
**exceeds the entire headline effect** (0.012).

"Only cell that regresses" is also a selection statistic that arises ~66% of the time under
a common improvement rate, and it holds only on mean@4.

**The frustrating part: the paper ignores its own stronger pre-registered statistic.** The
analysis contract specified val-AUC, final pass@8, and dead-fraction. On *between-cell*
step-50 numbers the effect is 2–4× the noise floor: mean@4 −0.027 (z=−2.02), pass@4 −0.048
(z=−1.98), val-AUC −12.5%. Report those and quantify the floor.

### M2. "The teacher was verifiably steering" is a min-vs-mean artifact

`PAPER.md:277` / `main.tex:325` / `GSM8K_ANALYSIS.md:42`: dead-sampled fraction "driven to
0.48 vs the ~0.65 population rate." I recomputed from the driver logs:

| cell | mean | min | n |
|---|---|---|---|
| maxrl **uniform** | 0.685 | 0.562 | 30 |
| maxrl **+teacher** | 0.643 | 0.484 | 50 |
| grpo **uniform** | 0.667 | 0.516 | 50 |
| grpo **+teacher** | 0.668 | 0.508 | 50 |

The 0.48 is a per-step **minimum**; 0.65 is a **mean**. Uniform's own min is 0.516 = 33/64 —
one prompt away, and `E[min of 50 draws of Binom(64,0.65)/64] = 0.513`. Like-for-like, the
GRPO arm's paired delta is **+0.001 (t=0.08)** — no steering at all in the very cell whose
regression is the paper's headline.

P-G4's pre-registered *control* also failed: uniform declined 0.744→0.602 at the same slope
(DiD t=−0.32), so the falling dead fraction measures the model improving, not allocation.
This is the same class of artifact already retracted once in `val_checkpoints.md`.

Reviewers found the one real difference between the arms: `CurriculumSampler` samples **with
replacement** while uniform uses `RandomSampler` without, so the teacher arm touched 34.4%
unique prompts vs 42.8% (matching `1−(1−1/7473)^3200 = 0.348` vs `3200/7473 = 0.428`). That
coverage penalty is an alternative explanation for the GRPO regression having nothing to do
with objective safety.

### M3. Step-matched, the pure teacher's maze gain nearly vanishes; only hindsight survives

The clock-matched protocol lets teachers run 1.18–1.37× more steps. I truncated every run to
its seed's uniform step count:

| config | ΔAUC clock-matched | ΔAUC step-matched | Δfinal-eval step-matched |
|---|---|---|---|
| frontier_alp (pure teacher) | +0.0097 (3/3) | **+0.0063** (t=1.17, p=0.36) | **+0.001 (1/3 seeds)** |
| champion (teacher + dense hindsight) | +0.0174 (3/3) | **+0.0173** (t=4.4–5.4, p≈0.04) | **+0.016 (3/3)** |

My paired test on the published AUCs: frontier_alp t(2)=2.06 (p≈0.18, CI [−0.011, +0.030]) —
**not significant**; champion t(2)=4.69 (p≈0.04) — significant. The "6/6 paired deltas
positive" framing is 2 correlated config families × 3 seeds, not 6 independent draws; the
honest sign-test floor at n=3 is p=0.125.

This **inverts the paper's channel-1-first ordering**: the robust maze effect is recycling,
not allocation. Throughput is a genuine benefit — report it in its own row instead of letting
it inflate an allocation claim.

### M4. The maze never tests the derived utility

`maze_gpu/train.py:87,141` implements `(1−(1−p)^N)(1−p)` — the **legacy heuristic**, not
`u_N = pass@N − p`. So §7.3's GPU evidence is not evidence for contribution #1. The CPU anchor
is the only rung where the derived form is the thing being measured, and there it is a **null
vs the retired heuristic** (−0.008 to +0.006, n.s.) while beating learnability p(1−p) by
+0.14 to +0.28. `THEORY.md:353-356` already says the honest version — "wins on principle
(parameter-free), not performance" — and the paper should adopt it.

Also: the maze runs the published p(1−p) baseline at **1 seed** vs 3 for frontier_alp, and its
seed-0 AUC (0.2273) ties frontier_alp (0.2328) under the doc's own ±0.01 rule.

### M5. Mass is never justified as the right objective, and it is beatable

No proposition argues that sampling ∝ u_N is decision-theoretically optimal (zero grep hits
for Fisher / SNR / expected-improvement anywhere in the repo). Two defects:

- **The docstring is literally false.** `teachers.py:163` and `verl_integration/curriculum.py:25`
  claim proportional sampling "maximizes expected learning signal per group." `Σ q_i u_i` is
  **linear** in q, so it is maximized at the vertex `q = δ_argmax`. Reviewers measured
  proportional captures 79.9% of achievable mass; γ=4 → 91.6%; γ=16 → 96.6%.
- **A variance-tilted utility beats it.** `(1−p)²·u_N` scores 0.7418 vs 0.7009 at γ=1
  (t=−8.76, 0/10 losses) — and as an *oracle with no hindsight* it beats the full stack
  10/10 paired seeds. The variance term peaks at p=0.0887, **left** of p*=0.1688: mass
  systematically over-weights the far frontier.

The repo's own V2 data already contains the refutation and doesn't draw it: Thompson collects
838 mass vs the oracle's 841 ("nearly saturated") yet AUC 0.700 vs 0.851. Mass-collected is
nearly uncorrelated with learning.

Getting ahead of this makes the paper *deeper*: derive the utility from a stated criterion and
show u_N is its leading term or a documented approximation.

### M6. `run_baselines.py` — the DAPO column is void

Verified CONFIRMED with a worse mechanism than reported: `weights_maxrl` returns all-zero
exactly when K=0 **or** K=N, so the uniform branch's `if np.any(w != 0)` guard (line 69) is
the *same predicate* as DAPO's `0 < k < n` keep-filter (line 59). The two arms produce
bit-identical draw sequences and final parameters (`|θ_u − θ_d|.max() == 0.0` in all 9
regime×seed cells). The eval-grid bug at line 89 (`if used % eval_every < 1`) then logs DAPO
at 2–6 points vs 9 for other arms, and AUC is the unweighted mean of logged points — so the
published ±0.09 DAPO effects are **pure logging artifacts**. Corrected: balanced DAPO 0.734
(= uniform exactly), easy-heavy 0.946 (= uniform exactly). The frontier-heavy 0.00 survives.

Note the audit passed these cells as "consistent-with-docs" *because* the script persists no
artifact (`AUDIT_ROUND2_REPORT_GUIDE.md:99`), while `PAPER.md:340` advertises that the audit
found only "prose qualifiers." A correct DAPO does exist in `frontier_rl/trainer.py` — port it.

### M7. Smaller but real

- **`ρ = −0.17` is not "the posterior LEARNS real difficulty."** It explains 2.8% of rank
  variance. At n=7449 the p-value (3.5e-17) says nothing about effect size.
- **"Up to 11×"** is one cell of a four-level table (1.2× / 2.7× / **0.5×** / 11×) — two of
  four reverse — and the 11× sits on a censored point (GRPO's L5 curve only reaches 0.25 at
  k=64). State the range or drop it from the abstract.
- **The 65–75% dead-group figure**: I confirmed the counter conflates K=0 with K=N
  (`weights_maxrl` returns zeros for both). It survives numerically — at the maze's final eval
  the all-pass share is only ≈0.08 of 8 groups vs all-fail ≈4.3 — but five different figures
  circulate across docs (65%/49%, 5.8→3.4, 5.2→3.9→2.6, 0.744→0.602). Two lines at
  `train.py:366` to log K==0 and K==N separately fixes it permanently.
- **Decay drift**: `teachers.py:49` hard-codes 0.9 with no constructor override while
  `frontier_rl/teacher.py:21` and `verl_integration/curriculum.py:52` use the validated 0.7,
  and `README.md:94` still advertises 0.9. The reproducibility claim here was **refuted** —
  every V1–V7 number does reproduce at its stated config — so this is hygiene, plus a
  `provenance:` line per table.
- **Bibliography is 8 entries.** That alone reads as desk-reject. The hindsight novelty claim
  ("unoccupied") needs CodeIt (ICML 2024 — relabels an ARC target to the realized program
  output), HER, R3, Minimo. A reviewer who knows CodeIt reads "unoccupied" as ignorance or
  concealment. The defensible delta: *exact-verifier goal relabeling of dead groups inside a
  policy-gradient RLVR loop with success-conditioned advantages, where the relabel is free
  because the verifier already ran.*

### M8. Two confounds I tested that came back clean (do not "fix" these)

- **γ=4 is not a Thompson-noise artifact.** I replaced the Thompson draw with the posterior
  mean: γ=1/2/4/8 → 0.7223 / 0.7343 / 0.7422 / 0.7623, tracking the Thompson arm
  (0.7072 / 0.7287 / 0.7481 / 0.7583). The compounding story survives.
- **Hindsight is not *only* an effective-lr increase, but the confound is partly real.** My
  controls (uniform teacher, 3 seeds, 400 steps):

  ```
  none        0.6562      <- baseline
  lr x2       0.8043      <- pure step-size increase captures most of the gap
  replay      0.8043      <- extra gradient on live groups, no new information
  randgoal    0.7897      <- relabel to a RANDOM prefix level
  hindsight   0.8780      <- full method
  fakelabel   0.4999      <- random rewards: WORSE than baseline (exactness matters)
  shallowgoal 0.7586      <- valid but suboptimal target
  ```

  Hindsight beats the lr/replay controls by +0.074 and beats random-target relabeling by
  +0.088, and breaking exactness is actively harmful. So relabel *direction carries
  information*. But ~67% of the headline +0.22 is reproduced by a control with **zero relabel
  information** — so "+0.22 from signal creation" is not the honest number. The
  extra-gradient placebo must ship with the claim.

---

## 5. Is the project in a productive loop?

Bluntly: **it is in a documentation loop.** Of the last 30 commits, 17 are doc/paper/audit/site
dominant. There are **27 markdown files against ~7k lines of Python**, including
audit-round documents auditing the project's own prose, a 350-line PAPER.md, a LaTeX paper, a
website, a 168-line READINESS.md, and consulting-response docs (SONIC_RESPONSE 16KB,
COSMOS3_RESPONSE 29KB) — while the flagship LLM experiment launched *today* with a config that
makes its central mechanism undetectable, and the one completed LLM experiment returned a null
on its positive prediction.

The audit loop's own coverage gap is the proof it should be automated: it audited 2 of 20
docs, passed `run_baselines.py` precisely because that script persists no artifact, and never
reached PAPER.md's abstract or MountainCar — **where the two worst defects live**.

The marginal doc is worth far less than the marginal GPU-hour right now. Replace ~1,300 lines
of prose audit with ~150 lines of pytest that recomputes every PAPER.md number from artifacts
and fails CI on drift.

---

## 6. Critical path — one A10G, 3–4 weeks

| # | Action | Cost | Kill criterion |
|---|---|---|---|
| **1** | **Kill cell C1 now.** Set `max_groups_per_step` 8→48+; regenerate the pool with random permutation + parenthesization + integer division (or use `Jiayi-Pan/Countdown-Tasks-3to4`); drop the 2-operand tier; add `str(old_target) not in response`; re-run the frozen probe | 0.5 GPU-h + 0.5 CPU-h (**saves 27–53 GPU-h**) | Post-SFT relabel yield stays <2% ⇒ hindsight can't ignite at 360M; re-scope or drop as flagship |
| **2** | **Complete the H6 2×2**: `frontier_grpo` seeds 1–2 + `uniform_maxrl` seed 0 with passk logging. Report the objective main effect with an exact permutation test | **2 GPU-h** | Interaction sign flips in any seed ⇒ demote to "objective main effect on coverage" (still p=0.0079, still publishable) |
| **3** | **The three missing CPU controls**: per-arm lr sweep; **extra-gradient placebo** (re-apply live-group gradients on dead groups, matched update count, zero relabel info); tilted-oracle + oracle+hindsight arms | **3 CPU-h, 0 GPU-h** | Placebo matches hindsight within 1 sd ⇒ thesis 2 is "gradient reuse," rewrite §5 and the abstract **before** any LLM spend |
| **4** | **Artifact remediation**: delete MountainCar 0.944/1.000 → branch artifact; fix `run_baselines.py:89` grid + DAPO accounting; commit `ksweep_results.json`; pytest that recomputes PAPER.md from artifacts | 3 CPU-h | None — but any number that can't be reproduced gets deleted the same day, not footnoted |
| **5** | **Rewrite §7.1/§7.2** around the controls: add uniform+hindsight and relabel-restricted arms; restate as "recycling discovers the sub-pool a sampler would have to be handed" and "hindsight adds +0.004–0.006 on top of the best sampler including an oracle" | 2 CPU-h | — |
| **6** | **Re-eval the 4 saved GSM8K checkpoints** with vllm at n=16 on all 1209 platinum rows; report the same-model noise floor as an explicit row; re-score P-G1 negative-on-its-own-metric, P-G4 untested | **2 GPU-h**, no retraining | Cross-k DiD loses consistent sign ⇒ §7.5 becomes a plumbing/diagnosis section, not an abstract claim |
| **7** | **Only if #1's probe passes**: reduced Countdown grid — 4 cells not 8 ({maxrl} × {hs on/off} × {teacher on/off}), 2 seeds, + a 2×-lr no-hindsight control. Add persistent logging (E-LLM-1's evidence had to be reconstructed from `/tmp/ray`) | 25–35 GPU-h | Tier-2 pass@16 <0.02 after 60 steps, **or** the 2×-lr control matches ⇒ ship on maze+CPU with an honest negative LLM rung |

Items 2, 3, 6 total **4 GPU-h + 6 CPU-h** and decide the paper's framing. They come before
the 30-GPU-h flagship because each can independently invalidate its premise.

## 7. Stop doing

- **Writing documentation.** Freeze PAPER.md / main.tex / README / EVIDENCE / REPORT / GUIDE /
  READINESS until a new artifact lands. Hard rule: no doc commit unless the same commit
  changes a file under `results/`.
- **The prose audit loop.** Replace with pytest (see #4).
- **Quoting the V1 cosines** (0.956/0.958) as Prop 6 evidence anywhere.
- **Adding adapters and consulting deliverables.** IsaacLab, Cosmos/LIBERO, pilot0, streaming,
  cosmos-live are ~1,300 lines appearing in **zero** results table. The ladder's problem is
  that existing rungs lack control arms, not that it needs more rungs.
- **Reporting matched-clock AUC without the step-matched number beside it.**
- **Saying "the oracle" or "ceiling."** If a ceiling is claimed it must be a sup over a
  *declared family*, and the family must include the tilt and no-floor variants.
- **Single-seed cells reporting orderings.** Two seeds and half the cells beats one seed and
  all the cells at the same cost — the project has already had to retract or soften two
  single-seed LLM artifacts.

---

## 8. The paper to write

**Write the safety paper, not the curriculum paper.**

> **Curricula Are Not Objective-Agnostic: Difficulty Sampling Amplifies the Advantage
> Estimator's Pathology**

Current draft as an ICML/NeurIPS/ICLR submission: **~10–15% accept** (elementary algebra
framed as a theorem; two headline empirical claims die to ten-minute ablations; §7.4 has no
artifact). Reframed with critical-path #2–#5 done: **~35–45%**.

Claim set, in evidentiary order:

1. **Lemma section (labeled elementary).** The expected advantage mass of a
   success-conditioned estimator has a closed form and differs qualitatively across
   estimators: u_N peaks at p* = 1−N^(−1/(N−1)); RLOO's mass is 2p(1−p) independent of N;
   GRPO's finite-N mass is throttled on near-dead groups. Include the honest note that
   `u_N = w_{N−1}(p)·p(1−p)` — MaxRL's own weight function times the base gradient. The
   contribution is the *sampling reading* and the compute-indexing by N, not the algebra.
2. **Therefore the same sampling rule has opposite sign under different objectives —
   predicted from the algebra, then measured.** Lead with the permutation test: 5 MaxRL arms
   gain pass@8 (+0.005 to +0.048), 4 GRPO arms lose it (−0.019 to −0.063), **zero exceptions**;
   objective main effect **+0.0664, t=5.81, exact permutation p=0.0079**. Assumption-free,
   which matters at n=9. I verified every one of those 9 deltas from raw logs.
3. **It replicates in sign at LLM scale** — the cross-k DiD (−0.0032/−0.0055/−0.0064/−0.0078),
   reported beside the measured same-model noise floor (0.0094/0.0172). Retire .096→.093.
4. **Practical rule: audit your estimator before you ship a curriculum.** This is the sentence
   practitioners will cite.

Demote the teacher to a §5 "and here is the sampler the algebra recommends," with the small
honest gains (ties the retired heuristic; beats learnability p(1−p) by +0.14–0.28) framed
exactly as THEORY.md already does. Include the tilted-oracle result as a stated limitation —
mass is not the optimal criterion, and the variance term peaks left of p*. Reviewers reward
that; they punish discovering it themselves.

**Move hindsight to paper 2.** It is the more novel idea and currently has one uncontaminated
number (+0.003–0.009 on top of an oracle sampler). Paper 2 needs the extra-gradient placebo,
a prompt-conditioned non-degenerate Prop 6 probe, and a Countdown run with the dose fixed.

---

## 9. The one-paragraph version

The theory is correct but elementary, and it is oversold as "theorem, not heuristic." The
allocation channel — contribution #1 — is worth +0.006 step-matched AUC on the only rung with
real gradients, is a null at LLM scale, and is not even the utility the maze code implements.
The creation channel — the novelty claim — has never been tested where it is both non-trivial
and non-confounded: its exactness probe cannot fail by construction, its categorical result is
task-pool access, and its ceiling-breaking result inverts when the oracle gets the same second
channel. The safety channel is the real contribution: it is predicted from the algebra,
replicates across 9 maze runs at permutation p=0.0079 with zero exceptions, replicates in sign
at LLM scale on a statistic immune to the noise floor, occupies open territory, and needs **2
GPU-hours** to finish. It is currently contribution 4 of 5, in §6–7. Lead with it. Then kill
the running Countdown cell, fix two config lines, and spend 4 GPU-hours + 6 CPU-hours on the
control arms before the 30-GPU-hour flagship — because three of them can each invalidate its
premise, and one of them (the extra-gradient placebo) decides whether the paper's centerpiece
mechanism is signal creation or gradient reuse.

---

*Method note: 7 parallel reviewers → 10 highest-severity findings through adversarial
refutation (9 survived, 1 refuted: the "V6/V7 numbers are unreproducible" claim — every
V1–V7 number does reproduce at its stated config) → synthesis. Reviewer overstatements were
corrected by the verification pass and are not reported here. My own independent
re-derivations, re-runs, and control experiments are marked "I measured / I ran" throughout.*
