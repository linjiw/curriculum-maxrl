# Round 5 review triage

**Review input:** `paper/main_iclr.pdf`, SHA-256
`37421c77c2d67631b8d0d9b97f33c0991c08b328324a0de6b6039972327497e7`  
**Panel result:** no mandatory `block`; the sprint contract nevertheless
returns `major_revision` because 4/5 reviewers had at least two mandatory
dimensions at `warn` or worse. The label is an ordinal contract output, not an
acceptance probability.

This triage follows the PI instruction to repair wording, scope, attribution,
and artifact disclosure without authorizing another experiment. The frozen P0
verdict and its four non-claims remain unchanged in substance.

## Accepted and implemented

1. **Aggregation sign.** The abstract, Corollary 2, OpenReview candidate,
   README, and website now state the one-shared-task-per-group,
   conditionally-i.i.d.-given-that-task mixture regime. They also state that a
   general dependent count law need not have the nonnegative sign.
2. **Expected-update bridge.** The scalar two-mean factorization now states its
   atomic conditional-i.i.d. and conditional-score-independence assumptions;
   no such factorization is claimed for an arbitrary dependent count law.
3. **Exponent sweep.** `N` is the deployed estimator group size and `M` is a
   tunable curriculum exponent. The Acrobot sweep rejects deployed-`N` peak
   specificity; it no longer licenses `M>=N` as a general deployment floor.
4. **P0 estimand.** The paper continues to report the frozen supported verdict
   while stating that score-policy substitution includes induced visitation,
   hardness, and exposure and does not establish activity mediation.
5. **Activity boundary.** “The theory supplies the utility” is now “theory
   supplies the activity score.” Full count-law activity represents outcome
   dependence; only the scalar pass-rate reduction assumes conditional i.i.d.
   The conclusion names Acrobot and the Digits own-unit counterexample, so unit
   matching is no longer written as sufficient for curriculum utility.
6. **Evidence status.** Acrobot is described literally as prospectively
   source-locked and confirmed. P0 remains preregistered and confirmed because
   its pre-run record is commit-bound. A study-by-study appendix table separates
   those labels and raw-data boundaries.
7. **Reproducibility perimeter.** Appendix D now distinguishes local
   theory/implementation verification, stored-summary/figure reproduction, raw
   endpoint reanalysis, and full training reproduction. P0's complete raw
   campaign is explicitly external.
8. **Inferential terminology.** The appendix defines “exact sign-flip” as
   exhaustive enumeration under sign exchangeability, not automatically
   design-based randomization inference.
9. **Attribution.** The original RLOO and GRPO papers are cited at the estimator
   definitions. The SFL variable-count statement names the audited Kinetix
   commit and source file in Appendix D.
10. **Presentation.** Figure 2 now says inside the panel that heterogeneous
    endpoint scales are direction-only; Figure 5 reports 10 correlation bins
    separately from 288,000 contributing group draws; the cross-platform
    p-value-magnitude comparison was removed; “come free” is narrowed to “no
    additional rollouts when grouped counts are retained.” Claim trace is
    97/97.

## Not adopted

- **Title change:** not made. The title was explicitly selected by the PI; the
  panel itself did not mandate a change. The abstract now states the atomic
  regime in which mean pass rate is sufficient.
- **P0 downgrade:** not made. Its immutable pre-run commit and terminal result
  support Tier 2. The paper instead distinguishes P0 from Acrobot and older
  maze provenance.
- **New robustness p-values or dispersion analyses:** not added in this
  wording/scope pass. They would be post-result analyses and are not needed to
  state the sign-exchangeability assumption accurately.
- **New mediation, multi-domain, LLM-RL, variable-`N`, two-stage curriculum, or
  named-baseline experiments:** rejected as outside this triage and contrary
  to the PI's “E4 only; no other experiments” decision.
- **Bandit/regret reframing:** not adopted. State freshness and coverage costs
  are now named, but the paper remains a boundary-mapped estimator-activity
  paper rather than claiming a nonstationary-bandit theorem.

## Acceptance checks

- The compact conclusion ends on page 9; references begin on page 10.
- No undefined reference, undefined citation, overfull box, fatal error, or
  emergency stop appears in the compact build log.
- The OpenReview candidate, README, website, and paper share the corrected
  aggregation scope.
- The exact reproduction, clean-clone, and anonymous-export receipts are
  recorded separately in the pre-submission checklist.
