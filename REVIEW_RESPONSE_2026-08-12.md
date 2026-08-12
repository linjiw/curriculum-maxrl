# Response to adversarial referee pass — 2026-08-12

**Reviewed object:** `paper/body_iclr.tex` compiled as `main_iclr2027.tex` (tectonic 0.17.0, provisional engine).
**Referee verdict before fixes:** 5 (marginally below threshold); modal outcome "reject with respectful reviews." Top risks: the novelty framing's self-contradiction, the estimator-matching claim doing no empirical work, and abstract readability.
**Disposition:** every major finding addressed by edit or explicit concession; minors either fixed or logged as accepted risks below. All edits preserved the frozen statistics, terminology contracts, and limitation disclosures; the conclusion still ends on page 9 after the changes.

## Fixed by edit

- **N1 (novelty self-contradiction).** The related-work sentence asserted the concurrent SD-branch quantities are "flat in N" two lines after quoting groupstd's N-dependent silent-group mass p^G+(1−p)^G. Rewritten: the SD branch's per-rollout magnitude peaks at p=.5 independent of N with N entering through the silent-group factor, versus our unnormalized pass@N-gap whose peak migrates as ln N/N. The "currently empty cell" rhetoric was replaced with "a distinct, testable functional form," plus an explicit sentence that superiority over the concurrent methods is untested and not claimed (also closes N3's honest-retreat demand).
- **N2 + A1 (estimator-matching vs rollout-awareness).** The Digits counter-test's own result (mismatched RLOO prefers the same sampler) means the estimator-matching pillar does no empirical work. Conceded explicitly: abstract now says "Rollout-aware selection carries empirical content" (was "the score's deployed-N shape"), Contribution 2 states "the supported content is rollout-aware difficulty targeting, not estimator matching," and Limitations adds that no arm pairs a mismatched-N score with a fixed deployed N, so peak-location specificity is untested.
- **N3 (probe-cost transfer relevance).** Added: in RLVR loops whose pass-rate posteriors update free of charge from training rollouts, probe accounting is a property of probing semantics, not of posterior-based selection.
- **E1 (scale billing).** Contribution 3 retitled "Coverage accounting on neural gradients" (was "at neural scale"); Evidence preamble now states the score's positive support is deliberately small-scale and the neural families test estimator-side predictions, not the score.
- **E2 (coined term from a 3-seed aggregate).** "Recycling-package sharpening" retired; subsection retitled "Countdown recycling: accuracy up, coverage proxy down"; the result paragraph now explicitly attaches no name or effect claim and frames the aggregate as demonstrating the ambiguity that motivates raw-outcome pass@k reporting.
- **S1 (paid-probe CI vs t discrepancy).** Added one sentence: the bootstrap interval excludes zero while the frozen t criterion fails, but the verdict does not hinge on that — the point estimate misses the frozen .02 SESOI fourfold.
- **S2 (maze sign-test granularity floor).** Acknowledged that 6/6 is the only outcome clearing .05 and the frozen 5/6 bar could not; inferential weight shifted explicitly to the block-averaged mean with leave-one-block-out intervals.
- **S4 (multiplicity).** Limitations now states the four families froze primaries independently at α=.05 and no paper-level error rate is claimed.
- **I1 (abstract vs body strength on Digits).** Abstract now attributes the rejection to the mismatched-RLOO result.
- **I2 ("beats" vs "supported by rule").** Contribution 2 aligned to "is supported over both … by the frozen rule."
- **I4 (conclusion sampler overreach).** Replaced with "the estimator-conditioned coverage ordering it predicted persisted under both tested samplers."
- **C1 (abstract readability + garbled clause).** Abstract rewritten; the "higher time-integrated MaxRL than GRPO coverage" clause fixed to "higher time-integrated coverage under MaxRL than under GRPO."
- **C2 (undefined primary metric).** Target-uniform normalized transition-AUC defined at first use.
- **C3 ("post-guidance" undefined).** Defined parenthetically at first Limitations use.

## Accepted risks (logged, not edited)

- **N4:** the truncation-order lemma remains in the abstract/Contribution 1; its delta over RL2ML is disclosed in Related Work. Judged worth its billing because the deployed convention is what every experiment runs.
- **E3 (scope-map table):** a settings × frozen-prediction × verdict table would help a fast reader but costs main-text space the 9-page bound no longer has; the Evidence preamble's family-to-claim map carries this. Revisit if reconciliation buys space.
- **C4 (p overloading), C5 (wave-narrative interleaving, Table 2 decision-column wording):** cosmetic; deferred to the pinned-environment pass to avoid churning frozen-verdict phrasing.
- **S3:** the Countdown aggregate keeps its abstract sentence — it motivates the operational recommendation and is now explicitly claim-free (E2 fix).

## Post-fix expectation

The three named submission-killers (N1, N2+A1, S1+C1) are resolved in the direction the referee said could reach a 6. The remaining structural ceiling is real and disclosed: no positive score result above 640 parameters until E2c lands, and no comparison against any concurrent 2026 selection method — both now stated in the paper itself rather than left for reviewers to discover.
