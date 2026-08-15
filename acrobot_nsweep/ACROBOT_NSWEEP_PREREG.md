# Preregistration — Acrobot score-exponent dose–response

**Status:** FROZEN 2026-08-15 before any confirmatory run of this study.
**Schema:** `curriculum-maxrl/acrobot-nsweep/v1`
**Relationship to the U64 campaign:** this study **subsumes and extends** it. The
U64 confirmatory campaign (Hopper jobs 9375605 + 9375630) was launched first and
remains the primary preregistered test of peak-location specificity. This study
was written and frozen **before any U64 result was inspected** — at freeze time
the Hopper campaign was still running and no analyzer had been executed.

## 1. Why a curve beats a single over-shooting arm

The U64 campaign asks one binary question: does u_16 beat u_64? That is enough
to falsify peak-location specificity, but it is weak evidence *for* it, because
a single over-shoot could lose for reasons unrelated to peak location.

The sharper question is the **shape** of performance as a function of the score
exponent while the deployed estimator is held fixed at N = 16:

- **H_peak** (the paper's claim): performance is maximized when the score's peak
  sits where the deployed N puts it, i.e. an inverted-U in `N_score` with its
  maximum at 16.
- **H_hard** (the deflationary alternative): only peak *hardness* matters, so
  performance is non-decreasing in `N_score` across the tested range, with no
  interior maximum at 16.

These predict different curves, not just different signs of one contrast.

## 2. Design

Deployed estimator is **N = 16 for every arm**. Only the sampling score changes.

| arm | score | peak p*_N |
|---|---|---|
| `uniform` | constant 1/8 | — |
| `u2` | u_2 = p(1-p) | .5000 |
| `u4` | u_4 | .3700 |
| `u8` | u_8 | .2570 |
| **`u16`** | **u_16 (matches deployed N)** | **.1688** |
| `u32` | u_32 | .1058 |
| `u64` | u_64 | .0639 |
| `u128` | u_128 | .0361 |

Eight arms x 20 paired logical seeds (**20000–20019**, the V2/U64 block) = 160
runs. All other constants are inherited verbatim from the sealed V2 tournament
and are byte-identical to the U64 campaign: `Acrobot-v1`, fixed eight-threshold
pool, task-blind H64 actor (640 parameters), practical dropped-group MaxRL
estimator at N = 16, Beta tracker (decay .7, floor .1, gamma 1.0), lr 3e-4, no
hindsight, 2,000,000-transition budget, evaluation every 100,000 transitions
with 32 episodes, evaluation seed base 1,000,000.

All eight arms of a given seed run **in one process on one machine**, so every
within-seed paired contrast is exact. Sources are the same vendored, V2-lock-
verified tree used by the U64 campaign (`acrobot_u64/vendor/`).

Metric: paired target-uniform normalized transition-AUC, the engine field
`auc_mean_pass_by_transitions`, exactly as the V2 analyzer defines it.

## 3. Primary estimands

Family of two confirmatory contrasts, Holm-adjusted at familywise .05, each
supported iff mean paired difference >= **+0.01** (SESOI) and the exact
two-sided sign-flip p (2^20 assignments) is <= .05 after adjustment:

| id | contrast | tests |
|---|---|---|
| **P1** | u_16 − u_128 | the deployed peak beats an 8x-harder peak (over-shoot) |
| **P2** | u_16 − u_2 | the deployed peak beats the learnability slice (under-shoot) |

**H_peak requires both P1 and P2 supported.** A maximum at 16 needs the curve to
fall away on *both* sides; either alone is consistent with a monotone curve.

## 4. Preregistered shape statistics (reported regardless of P1/P2)

1. **Arm-mean curve** over `N_score` ∈ {2, 4, 8, 16, 32, 64, 128}, with paired
   bootstrap CIs against `u16`.
2. **argmax_N** of the arm-mean curve. Under H_peak this is 16.
3. **Monotonicity check**: Spearman rank correlation between `N_score` and the
   arm mean over the seven scored arms. Under H_hard this is strongly positive;
   under H_peak it is near zero or negative because of the fall-off past 16.
4. **Neighbour contrasts** u_16 − u_8 and u_16 − u_32, descriptive, to show
   whether any interior maximum is sharp or flat.

Statistics 1–4 are descriptive and carry no decision. They are frozen here so
they cannot be selected after seeing the curve.

## 5. Interpretation, fixed in advance

| P1 | P2 | argmax | conclusion |
|---|---|---|---|
| supported | supported | 16 | **H_peak supported.** Deployed-N peak location matters; the confound is broken. |
| supported | supported | ≠16 | Both tails fall away but the maximum sits elsewhere; report the curve and claim only "an interior optimum near the deployed N". |
| not supported | supported | ≥32 | **H_hard.** Harder-is-better; peak-location specificity NOT supported. Report prominently. |
| either | not supported | any | u_16 does not beat the learnability slice at all; the paper's Acrobot result does not survive here and that is the headline. |

## 6. Relationship to the U64 campaign, and why this is not result-shopping

- This document was frozen while the U64 campaign was still running and **before
  any U64 or nsweep result was read**.
- The two studies share seeds 20000–20019, so the arms at `N_score` ∈ {2, 16, 64}
  are a **cross-platform replication** of the U64 campaign: Hopper runs on Intel
  Xeon Gold 6240R under Linux 4.18/el8, this study runs on a different Linux
  x86_64 host under 6.8. Agreement is evidence of robustness; disagreement is a
  finding about platform sensitivity and must be reported as such.
- Both studies use the identical frozen decision rule and SESOI.
- **Both are reported regardless of outcome.** Neither may be dropped, and
  neither may be relabelled as exploratory after the fact.
- If the two disagree on the u16−u64 sign, the U64 campaign — launched first and
  preregistered first — is the primary, and the disagreement is reported.

## 7. Execution and stopping rules

1. All eight arms of a seed run in one process; each writes one JSON atomically
   via a `.partial` rename.
2. No inferential quantity is computed until all 160 cells are present and valid.
3. The analyzer runs **exactly once** and refuses an incomplete matrix, any
   `.partial`, any non-confirmatory mode, and any campaign spanning more than one
   lock digest.
4. Failed cells are re-run from the identical lock and seed. Seeds are never
   substituted and the block is never extended.
5. Nothing in sections 2–5 changes after this freeze.

## 8. Limits

- One task family, 640-parameter policy, one deployed N. This maps the curve at a
  single operating point; it is not a general law.
- A maximum at 16 shows the deployed-N peak beats the tested alternatives; it
  does not prove u_N is the optimal score family.
- Seven exponents on a log-ish grid cannot resolve a maximum finer than the
  spacing between 8 and 32.
- `u2` is a common-scaffold comparator, not an implementation of ProCuRL, SFL,
  PLR, PAIRED, ACCEL, or ALP-GMM.
