# MountainCar Neural Curriculum-Transfer V1R2: Development Result

Status: **independently verified development NO-GO; confirmatory seeds were not
run and are not authorized**.

This note records the post-execution result without modifying the immutable
pre-run protocol or any of its eleven locked source files.  The protocol header
therefore remains a historical registration snapshot rather than a current
execution-status page.

## Locked execution

- Environment: Gymnasium `MountainCar-v0` with the eight registered nested
  position thresholds.
- Development seeds: `17000..17002`.
- Conditions: frontier shared H64, uniform shared H64, hardest-only shared H64,
  uniform disjoint-total H8x8, and uniform disjoint-active H64x8.
- Matrix: all 15 registered runs completed.
- Reserved confirmatory seeds: `18000..18019`; untouched and unavailable in the
  V1R2 runner.

SHA-256 evidence chain:

- development lock:
  `b5edbc33048a8d3a8d7dbb992a23178ddf8424dd3c5be3165c87e6dc42a50a5c`
- development artifact:
  `2e4803805009a3323307f6bdcfae17fb625008adb5361dc2310e414a19129180`
- independent verification report:
  `fdefc9e4ee2887953c341d2f44c44001bc336598b089dbdc8035175e430148a0`

The independent analyzer reports `all_checks_passed=true`.  It reconstructed
the sampler traces, uniform schedules, rollout accounting, evaluation samples,
common random numbers, transition AUCs, capacity identities, and no-hindsight
contract from the saved evidence.

## Predeclared feasibility decision

Development feasibility is **false**.  The sole failed Boolean gate is
`natural_dead_mixed_all_pass_regimes`:

| Group regime | Pooled count |
|---|---:|
| all-fail / dead | 1,932 |
| mixed | 474 |
| all-pass | 0 |

All other technical gates passed, including the exact matrix, task coverage,
nonzero updates, capacity controls, common random numbers, paired uniform task
schedule, and projected confirmatory runtime.  The absence of all-pass groups
means this development matrix did not exercise the saturated `p` near one
side that the registered coefficient-mass curriculum is intended to
downweight.  Under the frozen rule, that alone stops the study.

## Descriptive learning result

The hardest-goal pass curve and its primary AUC were exactly zero in all 15
runs.  Consequently, every registered development contrast on the primary
metric was exactly zero:

| Descriptive contrast | Mean hardest-goal AUC difference |
|---|---:|
| frontier shared - uniform shared | 0.000 |
| frontier shared - hardest-only shared | 0.000 |
| uniform shared - uniform disjoint-total | 0.000 |
| uniform shared - uniform disjoint-active | 0.000 |

The supporting mean-pass AUC differences were positive but small (`+0.00651`,
`+0.01198`, `+0.00547`, and `+0.00430`, respectively).  They are descriptive
development quantities only.  The protocol makes hardest-goal AUC primary, so
the supporting metric cannot rescue a degenerate primary outcome or authorize
a performance claim.

## Decision and lesson

Do not run the reserved confirmatory block under V1R2.  This experiment does
not show that the proposed curriculum wins or loses on the native MountainCar
goal; it shows that the frozen actor, optimizer, and 500,000-transition budget
failed to create primary-metric headroom for any method.  A later version must
use new development seeds and an outcome-blind adequacy rule to establish both
native-goal variation and all three estimator regimes before freezing a new
confirmatory protocol.  The untouched `18000..18019` block must remain
reserved unless a separately reviewed protocol explicitly reuses it.
