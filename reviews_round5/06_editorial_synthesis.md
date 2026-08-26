# Round 5 Editorial Synthesis

**Manuscript:** *Score the Count Law, Not the Mean Pass Rate: Estimator Activity for Curriculum Selection in Verifiable-Reward RL*  
**Target:** ICLR 2027  
**Panel:** EIC, methodology, domain, cross-disciplinary/perspective, and devil's advocate  
**Decision:** `major_revision`

This synthesis applies the supplied five-reviewer sprint contract mechanically. It does not average rubric scores or add a sixth review. The devil's advocate did not issue a balanced recommendation, but its dimension signals count under the contract. Revision triage is limited to existing evidence, theoretical and claim-scope correction, wording, citations, and artifact/provenance disclosure. No new experiment is required or authorized by this synthesis.

## 1. Mechanical score matrix

| Dimension | Priority | EIC | Methodology | Domain | Perspective | Devil's advocate |
|---|---|---:|---:|---:|---:|---:|
| D1 — Methodology rigor | mandatory | warn | warn | warn | pass | warn |
| D2 — Domain accuracy | mandatory | warn | warn | warn | pass | warn |
| D3 — Argumentative coherence | mandatory | pass | warn | warn | warn | warn |
| D4 — Cross-disciplinary relevance | high | pass | pass | pass | warn | warn |
| D5 — Writing and structure | normal | pass | pass | pass | pass | warn |

### Reviewer-local F2 predicates

The local predicate is “at least two mandatory dimensions are `warn` or worse.”

| Reviewer | Mandatory warn-or-worse count | Local predicate |
|---|---:|---:|
| EIC | 2 (D1, D2) | positive |
| Methodology | 3 (D1–D3) | positive |
| Domain | 3 (D1–D3) | positive |
| Perspective | 1 (D3) | negative |
| Devil's advocate | 3 (D1–D3) | positive |

The verified panel count is therefore **4 of 5 positive**. This includes the devil's advocate's contract signals despite its lack of a balanced accept/reject recommendation.

## 2. F0–F3 evaluation and decision

| Rule | Evaluation | Fired | Consequence |
|---|---|---:|---|
| F1 — any mandatory `block` | No reviewer assigned `block` to D1, D2, or D3. | false | `reject_or_major` is not triggered. |
| F2 — at least 4/5 reviewers have two or more mandatory dimensions at `warn` or worse | EIC, methodology, domain, and devil qualify; perspective does not. Count = 4/5. | **true** | `major_revision` |
| F3 — any D4 `block` | D4 contains three passes and two warnings, with no block. | false | No additional major-revision trigger. |
| F0 — all mandatory dimensions pass for all reviewers and no higher rule fires | Multiple mandatory warnings remain, and F2 fires. | false | `accept` is unavailable. |

**Binding decision: `major_revision`.** F2 is the only fired failure rule. The decision is not a rejection: there is no mandatory block, and every reviewer treats the exact count-law contribution as surviving the requested corrections. The major-revision outcome arises from the breadth of warning-level claim, provenance, and inference-calibration issues across four reviewers.

## 3. Consensus strengths

- **[CONSENSUS-4] Memorable and useful central abstraction.** All four balanced reviewers regard the same-mean/different-count-law counterexample as an effective motivation for treating estimator activity as a functional of the success-count law. The devil's advocate also expressly preserves the arbitrary-law mass identity and counterexample under its strongest rival narrative.

- **[CONSENSUS-4] Strong exact theory for the stated estimator conventions.** The balanced panel credits the arbitrary-law MaxRL identity, the conditionally-i.i.d. scalar reduction, the deployed `T=N-1` convention, explicit zero-stabilizer boundary, estimator-specific mass shapes, and separation of coefficient activity from gradient norm, SNR, direction, and learning utility. The domain reviewer particularly credits the fair fixed-`N` SFL/RLOO identity and the resulting narrower novelty perimeter.

- **[CONSENSUS-4] P0 is a well-isolated same-substrate treatment contrast.** The reviewers agree that holding the estimator, group size, posterior machinery, generator, budget, warm start, floor, and paired blocks fixed while changing the score functional is a strong design for the narrow question it asks. They also credit the treatment-delivery check and block-level analysis.

- **[CONSENSUS-4] Boundary mapping is a strength.** Acrobot, the exponent sweep, AMaze, MAZE-SCORE, Digits, allocation, and the inconclusive gate are not pooled into a generic success narrative. Negative, descriptive, post-hoc, and inconclusive results are generally labeled at their earned level. The devil's advocate agrees that this transparency preserves the diagnostic contribution even when prescriptive interpretations are narrowed.

- **[CONSENSUS-4] The paper is well organized for a dense nine-page argument.** The balanced panel finds the counterexample-to-theory-to-evidence sequence coherent, the theorem interpretations useful, and the main-text page boundary respected. The remaining presentation work is local rather than a request for wholesale restructuring.

## 4. Central warning clusters

There are no blocking clusters. The following warning clusters drive the F2 major revision.

### A. Scope the signed aggregation result wherever it is used

**Reports:** EIC, methodology, and domain; the perspective reviewer judged the body statement adequate but did not resolve the abstract wording.

The arbitrary-law MaxRL identity is unsigned with respect to a plug-in gap. The nonnegative “over-predicts” result requires the mixture regime in which one atomic task `X` is sampled once for a group and all `N` rollouts are conditionally i.i.d. given that shared `X`. The abstract currently states the direction for “a coarse unit” without that load-bearing condition. The domain reviewer also asks that the sampling order be explicit in Corollary 2 rather than left implicit.

**Required resolution:** add the shared-atomic-task/conditionally-i.i.d. condition to the abstract and every directional practical summary; state the one-`X`-per-group sampling order in Corollary 2; and keep the arbitrary-count-law identity separate from the mixture-law sign. A short note that under-dispersed or anti-correlated laws can reverse the sign would prevent the abstract from being read as an arbitrary-law theorem.

### B. Separate estimator-matched activity from a tunable score exponent

**Reports:** EIC, methodology, domain, and perspective unanimously; devil's advocate corroboration.

All four balanced reviewers reject the imperative “use `u_M` for `M` at least the deployed `N`” as a general deployment rule. When the deployed estimator has `N=16`, `u_16` is the estimator-matched activity curve; `u_64` is a harder-peaked member of the same family, not the deployed estimator's activity. The sweep shows that the tested optimum was not at deployed `N` and that performance was not monotone through the full sweep. It does not establish `M>=N` as a transferable floor.

**Required resolution:** define `M` as a curriculum hardness/exponent hyperparameter distinct from deployed group size `N`; report the tested Acrobot result as evidence against deployed-`N` peak specificity; remove the universal lower-bound language; and present exponent sweeping, if mentioned, as a substrate-specific tuning hypothesis rather than a theorem or deployment guarantee.

### C. Make the activity-to-update and activity-to-utility boundaries exact

**Reports:** domain and devil's advocate, with the remaining reviewers agreeing on the broader activity-versus-utility boundary.

The devil's advocate identifies a credible unstated assumption in the displayed expected-update factorization. The paper moves from an arbitrary joint group law to the claim that `E[g-hat|x]` factors through two marginal conditional score means. Under general dependent groups, score geometry may remain count-conditioned after conditioning on a member's own reward. No other reviewer declares the factorization false, so this synthesis does not do so either; it does require the assumptions to be made auditable.

The domain reviewer separately identifies two direct wording conflicts: “the theory supplies the utility” should say “activity score,” and the limitation that activity “ignores non-i.i.d. rollouts” conflicts with the arbitrary-count-law result. Full count-law activity can represent outcome dependence; what it omits is score-gradient geometry and downstream dynamics. The devil's advocate also notes that the conclusion's own-unit sentence is category-level even though Digits is an own-unit counterexample to superiority over uniform.

**Required resolution:** state the conditional-independence/exchangeability assumptions under which the scalar expected-update factorization holds, or give the appropriate count-conditioned expression for the arbitrary-law case; change “utility” to “activity score”; rewrite the non-i.i.d. limitation to distinguish the full count law from the scalar `p` reduction; and make the conclusion explicitly Acrobot-specific before naming Digits as the boundary showing that correct granularity is not sufficient for learning utility.

### D. State the P0 estimand without borrowing an unproved mechanism

**Reports:** devil's advocate; the EIC, methodology, domain, and perspective reviewers regard the existing no-mediation caveats as a strength.

The panel agrees that P0 identifies the causal effect of substituting one scoring policy for another on the tested substrate. The disagreement is whether “causal relevance of the correction” implies that improved coefficient-activity calibration mediated the learning effect. The manuscript already says that Corollary 2 did not predict the learning sign and that the per-level secondary is not mediation evidence, but the imperative takeaway and some correction language still permit a mechanistic reading.

**Editorial resolution:** retain the causal treatment claim, but define it precisely: substituting the shared-posterior count-law scoring rule for the i.i.d.-at-the-mean plug-in improved cov-AUC on this substrate. State in the same paragraph that activity mediation is unsupported and that visitation, hardness, and exposure changes are part of the treatment package. This wording resolves the issue without a new mediation experiment and without weakening the registered treatment contrast.

### E. Reconcile evidence labels with study-by-study provenance

**Reports:** EIC, methodology, domain, and perspective unanimously; devil's advocate corroboration.

The contribution list's “registered and confirmed” label for Acrobot conflicts with “internally frozen” in the evidence section and with the limitation that registration timing is incomplete. Reviewers also find the phrase “maze records” too broad to determine whether P0's freeze timing and raw reconstruction are fully auditable. The problem is inconsistent entitlement language, not an allegation that the reported effects are fabricated or that P0's design is invalid.

**Required resolution:** add one compact provenance table with, for each load-bearing study, the evidence label, freeze object and time, immutable identifier/commit if available, first-run boundary, analyzer/result object, independent replicate, and raw-data availability. Keep “preregistered/confirmed” only where an independently auditable pre-execution record supports it. Otherwise use a literal lower-tier label such as “internally frozen and supported under its prespecified rule.” Do not apply a blanket downgrade to P0 if its existing immutable history supports the stronger label; instead make that evidence explicit.

### F. State the artifact boundary at the level of the claims it supports

**Reports:** EIC and methodology directly; domain, perspective, and devil's advocate corroborate the provenance boundary.

The one-command path verifies stored summaries, manifests, tests, and figures, but reviewers could not infer that it reconstructs every headline endpoint from raw runs. In particular, the methodology review identifies the external P0 2x48 raw campaign as missing from Appendix D's explicit external-data list. This matters because P0 is the causal closure, while summary verification and raw reanalysis are different reproducibility claims.

**Required resolution:** provide a single matrix distinguishing (1) local theory/implementation verification, (2) figure regeneration from stored summaries, (3) raw endpoint reanalysis, and (4) full training reproduction. Explicitly list P0 raw endpoints, telemetry, receipts, and their location or external checksum boundary. Vendor or anonymously archive existing raw records if feasible; if not, state plainly that the local command verifies the stored P0 summary but cannot reconstruct the primary from raw executions. Do not imply that missing checkpoints or outcomes can be regenerated locally.

### G. Calibrate “exact” inferential language

**Report:** methodology, at confidence 5/5.

The methodology reviewer distinguishes computational enumeration from design-based exactness. Pairing, common random numbers, and counterbalanced execution order do not by themselves establish arm-label exchangeability. The paper itself acknowledges this limitation for Acrobot.

**Required resolution:** name the stochastic or randomization assumption supporting each sign-flip test. If arm labels were not randomized under a sharp null, describe the result as exact enumeration under a sign-exchangeability assumption rather than unqualified “exact” inference. Existing block-level outcomes may be used for a sign-test or other sensitivity if already retained, but this synthesis does not require new data collection. Also report paired dispersion where it is already recoverable from the retained block records.

### H. Repair domain attribution and source-level claims

**Report:** domain.

The domain reviewer requests the original RLOO and GRPO sources at their first definitions and a versioned code citation for the claimed SFL variable-`n` implementation correction. The paper citation supports SFL's published algorithm; an implementation-specific statement needs an implementation-specific source. A classic intermediate-difficulty goal-curriculum antecedent is also suggested.

**Required resolution:** add the original RLOO and GRPO citations identified in the domain report, attach the variable-`n` statement to the exact audited SFL release/commit, and distinguish published-algorithm precedent from shipped-code behavior. Preserve the current, accurate fixed-`N` SFL/RLOO proportionality and the narrow surviving novelty wedge.

## 5. Secondary warnings and presentation repairs

- **Online partial-feedback contract (perspective; devil corroboration):** add a compact deployment note naming the grouping key, fixed- versus variable-`N` state, update timing, decay/floor semantics, minimum visitation, and telemetry for stale or sparse count-law estimates. Replace “come free” with “require no additional rollouts when grouped counts are already retained.” State that zero coefficient activity is not zero scientific, coverage, or stakeholder value and retain an exogenous coverage floor or separate coverage objective.

- **Figure 2 (EIC, methodology, domain):** put “direction only; magnitudes incomparable across rows” inside the panel or separate endpoint families visually. Do not rely solely on the caption.

- **Figure 5 (methodology):** report the number of bins used for `r=.90` separately from the 288,000 contributing group draws.

- **Cross-platform p-values (methodology):** delete the claim that portability checks have smaller p-values than V2. Compare effects and intervals, and retain the statement that shared seeds do not create new independent cohorts.

- **Terminology and notation (EIC, methodology, domain, perspective):** use one evidence-status typography; keep “coefficient activity” as the primary term; distinguish the estimator denominator stabilizer from the sampling floor; define `M` as a score exponent; and move raw implementation identifiers or full hashes to the appendix/manifest where needed for readability.

## 6. Disagreements and editorial resolutions

| Issue | Positions | Resolution |
|---|---|---|
| Is the core paper fundamentally unsound? | All balanced reviewers find a genuine contribution; the devil's advocate explicitly preserves the exact diagnostic even under its rival account. | No. There is no contract block. Revise the prescription and evidence labels while preserving the exact count-law contribution. |
| Is the aggregation-gap scope already adequate? | Perspective passes D2 based on the body theorem; EIC, methodology, and domain warn because the abstract omits the shared-task mixture condition. | The body does not cure an unqualified abstract. Add the condition to front matter and directional summaries. |
| Does P0 prove a mechanism? | Balanced reviewers praise the explicit no-mediation caveat; the devil argues that “correction” language still borrows mechanistic force. | Preserve the causal score-substitution effect; explicitly deny activity mediation and name induced visitation/hardness as part of the treatment. No new mediation experiment. |
| Must the title change? | EIC calls it memorable and faithful; the devil finds the imperative broader than the atomic regime, with the perspective reviewer noting the same risk. | No automatic title change is mandated. The title remains an author/PI decision, but the abstract must say when mean pass rate is sufficient and the deployment prose must lose its universal imperative. A qualified title remains an available editorial choice. |
| Is cross-disciplinary transfer a major empirical gap? | EIC, methodology, and domain pass D4; perspective and devil warn about partial feedback, nonstationarity, coverage, and variable-`N` deployment. | Add one bounded systems/coverage contract from existing definitions and logs. Do not expand the paper into a bandit theorem or require a transfer experiment. |
| Should the expected-update bridge be treated as wrong? | Only the devil raises the count-conditioned score issue; other reviewers accept the broader theoretical package. | Do not declare the result wrong by vote. Require explicit assumptions or an arbitrary-law expression so readers can verify the factorization's scope. |
| Does “exact sign-flip” require revision? | The methodology expert raises it; the other reports do not. | Follow the specialist warning: clarify exchangeability/randomization semantics and use existing block records for sensitivity only if available. No new run is needed. |

## 7. Bounded revision roadmap

### Priority 1 — Must fix before resubmission

| # | Action | Sources | Acceptance check |
|---|---|---|---|
| R1 | Add the one-shared-`X`-per-group, conditionally-i.i.d.-given-`X` assumption wherever the aggregation-gap direction appears. | EIC, methodology, domain | No unqualified “over-predicts” claim remains for arbitrary count laws. |
| R2 | Separate estimator group size `N` from score exponent `M`; delete the `M>=N` floor prescription. | All four balanced reviewers; devil | The sweep is described as a tested Acrobot boundary, not a general deployment law. |
| R3 | Audit the expected-update factorization and revise the “utility,” non-i.i.d., and own-unit conclusion sentences. | Domain, devil | Arbitrary-law mass and any conditional-i.i.d./score-independence bridge have distinct assumptions; the conclusion names Acrobot and the Digits boundary. |
| R4 | State P0's precise treatment estimand and repeat that coefficient-activity mediation is unsupported. | Devil; bounded by caveats praised by all balanced reviewers | The causal claim refers to score-policy substitution on one substrate, not a general activity mechanism or superiority to `p(1-p)`. |
| R5 | Reconcile “registered,” “preregistered,” “internally frozen,” “supported,” and “confirmed” study by study. | All reviewers | Every strong status label resolves to an explicit immutable pre-execution object, or is downgraded to literal auditable wording. |
| R6 | Add a four-level reproducibility matrix and explicitly disclose the P0 raw-data boundary. | EIC, methodology, domain, perspective, devil | A reader can tell what runs locally from summaries, what can be reanalyzed from raw outcomes, what requires external data, and what is blocked. |
| R7 | Qualify sign-flip “exactness” with its exchangeability/randomization basis and add retained-data descriptives/sensitivity where available. | Methodology | Inferential labels match their assumptions; no new experiment is introduced. |

### Priority 2 — Should fix in the same revision

| # | Action | Sources | Acceptance check |
|---|---|---|---|
| R8 | Add original RLOO/GRPO citations and a versioned source for the SFL implementation claim. | Domain | Method definitions cite primary sources and paper-versus-code claims are separated. |
| R9 | Add a compact online state, freshness, cost, and coverage contract; narrow “comes free.” | Perspective, devil | Deployment prose states the retained statistics, sampling feedback loop, coverage safeguard, and no-extra-rollout—not no-estimation-cost—claim. |
| R10 | Repair Figure 2/5 labels, remove cross-study p-value comparison, and normalize status/notation typography. | EIC, methodology, domain, perspective | Heterogeneous endpoint magnitudes cannot be mistaken as comparable; correlation sample size and independent-unit language are explicit. |

These ten items are deliberately bounded. Most replace or qualify existing prose. The provenance and reproducibility tables may live in the appendix with one main-text pointer so the conclusion remains within page 9.

## 8. Requests not adopted in this triage

The following reviewer suggestions are not conditions of this decision because they require new experiments or would expand the paper beyond its demonstrated contribution:

- a new multi-domain or LLM-scale validation of `M>=N`, variable-group-size deployment, or general RLVR transfer;
- a new mediation intervention matching difficulty, visitation, entropy, or exposure across score arms;
- a new two-stage curriculum experiment that gates on activity and ranks live units by a separate utility/progress score;
- new named-method superiority tests against full SFL, ProCuRL, PLR, or `p(1-p)` implementations;
- a new nonstationary-bandit regret theory or a wholesale reframing as an active-learning/bandit paper;
- reconstruction by rerun of unavailable checkpoint or outcome payloads solely to satisfy this review round.

These are legitimate future-work directions, not defects that must be repaired now. For the present submission, the correct response is narrower scope, precise causal and inferential language, and exact disclosure of what the existing artifact can and cannot establish.
