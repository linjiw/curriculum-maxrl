# ICRA 2027 navigation campaign — preregistration draft

**Status:** Outcome-blind protocol draft, 2026-08-11. The full BARN/Jackal
backend has not run. Freeze the remaining asset/container placeholders and
commit this file plus the split manifest before the first full seed finishes.

## Claim under test

At a deployed group size of at least 8, sampling navigation environments by
the estimator-derived utility

`u_N(p) = (1 - (1-p)^N) - p`

improves time-integrated, target-uniform success on a fixed held-out course set
relative to uniform sampling, compute-blind `p(1-p)` learnability, and a
hand-ordered difficulty curriculum.

The factor of two in expected total absolute advantage is immaterial to the
sampling distribution. The utility is hyperparameter-free conditional on N;
posterior decay and the uniform replay floor are teacher stability settings,
not part of that claim.

## Domain and split

- Primary domain: BARN-style lidar navigation with a shared lidar-to-velocity
  policy; Jackal hardware validation is optional and not a submission gate.
- Course record fields: immutable environment ID, scalar published difficulty,
  and asset path/checksum.
- Freeze a difficulty-stratified 80/20 train/held-out split with
  `freeze_pool_split.py --seed 20270811 --n-strata 10`.
- The held-out IDs, reset seeds, initial states, and success verifier are fixed
  across arms and never used for teacher updates.
- Required before launch: **BARN asset manifest SHA-256 = TBD** and
  **training container/image digest = TBD**.

## Arms and controls

Four primary arms, at least five paired seeds each:

1. `ours_uN`: decayed Beta posterior, Thompson sample, `u_N`, 10% uniform floor.
2. `uniform`: uniform environment sampling; posterior logged diagnostically.
3. `learnability`: identical posterior/Thompson/floor, utility `p(1-p)`.
4. `staged`: published difficulty order, uniform over the unlocked prefix,
   promotion at posterior success 0.7 after at least five frontier groups,
   with the same 10% floor.

All arms use the same policy, optimizer, estimator, reset distribution, group
size, number of environment interactions, evaluation schedule, and seed list.
Do not label arm 3 “ALP-GMM”: ALP-GMM uses temporal learning progress and a
mixture model. PLR may be added only as a separately frozen optional arm.

## Primary endpoint and budgets

- Primary: area under held-out target-uniform mean-success versus training
  wall-clock, evaluated at a common predeclared wall-clock grid on exclusive,
  like-for-like hardware.
- Co-primary accounting view: the same AUC versus environment transitions.
- Report both even when conclusions agree. If exclusive hardware cannot be
  guaranteed, transition-matched is primary and wall-clock is descriptive;
  this convention must be changed here before unblinding.
- Evaluation time is excluded from training wall-clock and reported separately.

## Secondary endpoints

- final held-out target-uniform success;
- success and time-integrated success by frozen difficulty decile;
- easy-decile retention, where “easy” is fixed from course metadata before runs;
- training episodes, simulator steps, and steps per GPU-hour;
- all-fail group rate and all-pass group rate;
- teacher posterior calibration against held-out evaluation, diagnostic only.

## Mandatory ablation

Run `N in {2, 4, 8, 16}` with the rollout budget matched. The confirmatory
score-shape comparison is deployed `u_N` versus `p(1-p)` at each N. This
ablation tests whether the compute-indexed peak is load-bearing; it is not
replaced by the existing Acrobot result.

## Analysis

- Independent unit: training seed. Environment courses and repeated samplers
  within a seed are repeated measurements, not independent replicates.
- Pair arms by training seed and fixed evaluation stream.
- Report paired mean delta, 95% paired bootstrap interval, every seed delta,
  positive/tie counts, and exact two-sided sign-flip p-value.
- No paper-level significance claim from fewer than five paired seeds.
- August 24 go/no-go: continue the ICRA deadline only if `ours_uN` is
  directionally at least as good as both uniform and learnability on the full
  domain. Otherwise preserve the campaign for RA-L and stop deadline-driven
  expansion.
- Analysis implementation: `icra2027/analyze_campaign.py`.
  **SHA-256 = `4017958334fad6db74d594d2442ba18bffadd72adfd0412502875b5f467efdf3`**.

## Scope and stopping rules

- Grid-navigation smoke results are engineering checks, never paper evidence.
- Do not inspect partial full-domain endpoints to tune score shape, split,
  promotion threshold, or seed set.
- Do not add PLR, HER, or hardware arms until the four-arm five-seed matrix is
  complete or safely running.
- Do not use the shared RTX 5090 while the frozen E2c occupancy gate is pending.
- BARN integration failure by August 17 activates the already-developed Isaac
  Lab fallback; failure of the August 24 directional gate activates RA-L.

## Venue constraints verified 2026-08-11

The official ICRA 2027 call states September 15, 2026, 11:59 PST and an
**8-page total limit including references**, double-column and
double-anonymous. This corrects the earlier six-pages-plus-references planning
assumption. AI-generated article content must be disclosed in acknowledgments;
review material may not be processed through an AI system.
