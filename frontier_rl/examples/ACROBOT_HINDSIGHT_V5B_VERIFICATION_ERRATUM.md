# Acrobot Hindsight V5B Verification Erratum

Status: **post-hoc, non-authorizing diagnostic**.

This erratum is bound only to the sealed V5B artifact with SHA-256
`c633886a121906ee2bceb03f3117e4bea5dc20ab314e43f9b702ef8d88f495ac`.
It does not amend the source lock, repair the registered analyzer, authorize the
registered primary family, or report or interpret any primary outcome.

## What failed

The original locked analyzer fails reproducibly with:

```text
runner V5B case diagnostic summary mismatch: lr_mult_0p5_hs_0
```

The mismatch is confined to exact equality of diagnostic cumulative step-norm
reductions. The runner saved these fields using NumPy `sum` and `dot`. The
analyzer reconstructed them using sequential Python `+=` and then compared the
nested diagnostic dictionaries exactly, bypassing its otherwise-used absolute
tolerance of `1e-12`.

## Forensic result

The read-only verifier checks the complete 9-cell by 20-seed artifact with the
frozen protocol, raw-run, source/runtime-lock, and linked-authorization helpers.
It validates 180 runs, 53,510 group records, 45,000 applied-update records, and
1,080 evaluation checkpoints.

This is not a wholly independent reimplementation. The forensic verifier
imports the frozen analyzer and reuses its private lock, raw-run, and
reconstruction helpers; its separate contribution is the reduction-order
diagnosis, byte-bound artifact check, and outcome-free report boundary.

Across the 720 saved cumulative diagnostic reductions:

- all 720 exactly equal a fresh reconstruction with the runner's NumPy
  reduction semantics;
- 377 differ bitwise from the analyzer's sequential-Python reductions;
- the maximum absolute difference is
  `1.9984014443252818e-15`;
- the maximum distance is 11 ULP; and
- every difference is within the frozen analyzer's `1e-12` numeric tolerance.

As a bounded diagnostic only, an in-memory copy was normalized to the
analyzer's sequential-Python diagnostic reductions. All remaining frozen checks
then passed. The normalized copy was not written, and primary contrast,
cell-outcome, and decision subtrees were neither retained nor emitted by the
forensic report.

## Artifact-generation protocol violation

The frozen protocol says that if any numeric verification fails, no member of
the primary family is computed or claimed. The sealed raw artifact nevertheless
contains runner-produced `stage_b_case_summaries`, `paired_scale_contrasts`,
`predeclared_scale_decision`, and `analysis_status.performed=true` fields. The
prose and forensic report make no outcome claim, but the literal "not computed"
rule was already violated when those subtrees were written before independent
analyzer acceptance.

Those fields are quarantined: they are not copied into the forensic report,
reported, interpreted, or eligible for retrospective authorization. A V5C
runner must keep raw execution records separate from primary analysis output
and emit the latter only after its frozen verifier accepts the artifact.

## Claim boundary

`registered_primary_family_authorized = false` and
`original_locked_analyzer_passed = false` remain the controlling conclusions.
This evidence identifies a reduction-order verification defect; it must not be
used to rescue, interpret, or publish the V5B registered primary results.
