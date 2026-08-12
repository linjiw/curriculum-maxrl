# Acrobot fixed-pool curriculum tournament — V2 results

Status: **complete; registered primary confirmed** (2026-08-08).

This report records the result of the source-locked V2 tournament defined in
`ACROBOT_CURRICULUM_TOURNAMENT_PROTOCOL.md`. The raw runner wrote no
inferential results. Analysis began only after all 60 confirmatory runs had
completed, with 20/20 valid runs in each arm and an empty failure ledger.

## Question and claim boundary

The experiment asks whether the rollout-aware practical-MaxRL score

\[
u_{16}(p)=1-(1-p)^{16}-p
\]

improves fixed-target learning efficiency relative to the common
\(p(1-p)=u_2(p)\) score when the learner actually uses \(N=16\) rollouts.
Uniform sampling is a multiplicity-controlled secondary comparator.

All three arms use the same official Gymnasium `Acrobot-v1` dynamics, fixed
eight-threshold task pool, shared task-blind H64 actor (640 parameters),
practical dropped-group MaxRL estimator, discounted Beta tracker where
applicable, \(N=16\), learning rate \(3\times10^{-4}\), no hindsight, and
nominal two-million-transition budget. Only the task-selection score changes.
The \(p(1-p)\) arm is a common-scaffold score comparator, not a full
implementation of ProCuRL, SFL, PLR, PAIRED, ACCEL, or ALP-GMM.

## Registered result

The primary estimand is paired target-uniform normalized transition-AUC for
\(u_{16}-p(1-p)\), over fresh logical seeds 20000--20019. The frozen decision
requires both a mean difference of at least \(+0.01\) and an exact two-sided
paired sign-flip \(p\le .05\).

| registered comparison | arm means | paired mean difference | paired bootstrap 95% CI | exact \(p\) | multiplicity | decision |
|---|---:|---:|---:|---:|---:|---|
| \(u_{16}-p(1-p)\) (primary) | .68711 vs .63907 | **+.04803** | **[+.02094, +.07385]** | **.003361** | single primary | **confirmed by frozen rule; estimate clears +.01 filter** |
| \(p(1-p)-\)uniform (secondary) | .63907 vs .64523 | -.00616 | [-.02264, +.01220] | .507843 | Holm-adjusted .507843 | not supported |
| \(u_{16}-\)uniform (secondary) | .68711 vs .64523 | **+.04187** | **[+.02182, +.06059]** | **.000809** | **Holm-adjusted .001617** | **supported** |

The primary paired signs are 15 positive and 5 negative. The two secondary
paired-sign counts are 7/13 for \(p(1-p)-\)uniform and 17/3 for
\(u_{16}-\)uniform. The exact \(p\)-value tests a zero paired contrast, not
the \(+0.01\) smallest effect of interest; passing both decision filters does
not statistically establish that the population effect is at least \(+0.01\).
The sign-flip interpretation additionally requires sign exchangeability under
the sharp null.

The result has the protocol's **P+/U+** interpretation: in this one
fixed-pool, common-learner setting, the deployed-\(N\) score beats both its
\(N=2\) score ablation and uniform sampling. It does not show that
\(p(1-p)\) is generally harmful, nor that FrontierMax beats a tuned or native
implementation of a named curriculum algorithm.

## Predeclared descriptive diagnostics

The following endpoints were declared descriptive and cannot rescue the
primary:

| endpoint | uniform | \(p(1-p)\) | \(u_{16}\) | paired \(u_{16}-p(1-p)\), 95% bootstrap CI |
|---|---:|---:|---:|---:|
| native-success transition-AUC | .31085 | .30521 | .37615 | +.07094 [+.03810, +.10132] |
| native-return transition-AUC | -466.81 | -466.08 | -457.42 | +8.66 [+3.84, +13.20] |
| target-uniform sampled-group AUC | .66268 | .65004 | .70299 | +.05294 [+.02549, +.07987] |
| target-uniform optimizer-update AUC | .65281 | .64598 | .69965 | +.05366 [+.02819, +.07777] |
| sampled groups / million transitions | 172.36 | 157.83 | 155.43 | -2.40 [-4.31, -.56] |
| optimizer updates / million transitions | 129.15 | 138.38 | 134.96 | -3.42 [-4.79, -2.11] |
| coefficient mass / group | .61503 | .73439 | .74904 | +.01465 [-.01911, +.04995] |
| coefficient mass / million transitions | 105.63 | 115.57 | 115.90 | +.33 [-3.61, +4.44] |
| nonzero-mass group fraction | .75018 | .87733 | .86909 | -.00823 [-.01969, +.00371] |
| final native success | .61875 | .66875 | .71563 | +.04688 [-.00938, +.10160] |

The mean \(u_{16}-p(1-p)\) contrasts remain descriptively positive on the
sampled-group (\(+.05294\)) and optimizer-update (\(+.05366\)) axes.
\(u_{16}\) also receives 2.40 fewer groups and 3.42 fewer updates per million
transitions, so it did not win by receiving more updates; these
non-confirmatory axes do not identify a causal mechanism. Observed mass means
are close between adaptive arms (.7344 versus .7490 per group; 115.57 versus
115.90 per million transitions), and both descriptive paired intervals
include zero; no equivalence test was registered.

## Integrity and accounting

- Exactly three arms by 20 logical paired seeds completed; every raw run
  passed numerical, accounting, verifier, parameter-count, cadence, curve,
  evaluation-state, and RNG-contract checks.
- Actual transitions include the last complete \(N=16\) group. Overshoot
  ranges were 262--5,830 (uniform), 102--4,194 (\(p(1-p)\)), and
  495--6,618 (\(u_{16}\)), all within the frozen 8,000-transition maximum.
- The fresh V2 seed blocks use globally disjoint domain roots. The three arms
  intentionally share roots within each logical seed as paired common random
  numbers.
- The portable verifier reopens the development raw/gate, rechecks their
  hashes, validates every confirmatory ledger, and deterministically
  reproduces the stored analysis. It is not a second training run and reuses
  the locked analyzer's statistical functions.
- Evaluation records retain aggregate nested success curves and checkpoint
  coordinates, not a per-episode trajectory ledger; individual evaluation
  trajectories cannot be reconstructed.

## Scope and next experiment

This result isolates a utility score in one fixed eight-threshold Acrobot
family and one 640-parameter learner. It does not test held-out-task
generalization, another estimator, a separately tuned optimizer, a full named
curriculum implementation, or a procedural environment generator. The
20-seed choice was inherited from historical V3 and was not selected by a
prospective power calculation for the new primary.

The result supports including Acrobot as a compact classical-control
confirmation of the arbitrary-\(N\) score shape. The next Mac-only named-method
study, if needed, is a separately frozen native ProCuRL-env softmax sampler
with every success-estimation probe charged. A faithful PAIRED/ACCEL or full
SFL UED comparison belongs in a procedural-level benchmark with held-out
transfer. A fixed-pool PLR adaptation would instead require its own
critic/value-error and staleness implementation plus a charged budget
contract.

## Frozen artifacts and hashes

| artifact | SHA-256 |
|---|---|
| V2 source/runtime lock | `0e6438d42ddc53b89d774233805c465dc562bb6be5f8ac93ecf8a4d09b5d9af3` |
| development raw | `c616912569f4d19e36ea4a8685616a35bef037934e5c8d366ee7bd51bb2c3311` |
| development gate | `6dc908e22e874550e0536f1fcd52f2b3a1768d1a89c510275bef7efc2e2baac6` |
| confirmatory raw | `f533d0b84cdb3f7d3ede4bc4c94aa11e3b0ffc58c8bc7ea1a26491476873b2c6` |
| locked analysis | `463fa1a01d95922976f09f75b21f6d8f2c6a8d256081ebedfa4ba968a06f356b` |
| post-lock portable verifier source | `12b0f636bb8904d04c4243c1d7ac3feb8d00615b78a86f067b1a15fbdb5d0e55` |

The source lock is local repository evidence. Its hash establishes consistency
with the supplied lock; independent pre-execution timing requires a trusted
copy or repository history. The portable verifier was added after the source
lock and is reported separately rather than described as pre-execution locked.
