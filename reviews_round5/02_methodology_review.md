## Dimension Scores

### D1: methodology_rigor

**score:** `warn`

**evidence:** The central P0 design is unusually disciplined for computational RL: it changes one score functional, uses 48 paired seed/warm-start blocks, declares the block as the independent unit, holds the substrate and budget fixed, checks treatment delivery, requires a complete 2x48 matrix, and reports the paired mean, 95% interval, exact test, positive-pair count, and SESOI (PDF p. 7, Section 3.4). The frozen protocol additionally documents prospective power, counterbalanced run order, a single confirmatory contrast, immutable hashes, and fail-closed missingness. The theory is explicit about zero-stabilizer conventions and arbitrary-law versus conditional-i.i.d. assumptions (pp. 3-5, Section 2; pp. 12-13, Appendix A). However, important localized gaps meet the precommitted warn threshold: Acrobot's 20-seed sample had no prospective power calculation (acknowledged on p. 17), its pre-execution timing is supported only by a local lock rather than an independently trusted registration, and the sign-flip inference relies on sign exchangeability that pairing alone does not establish (p. 17). P0's raw 2x48 campaign remains external, while Appendix D's missing-data list does not explicitly name that central boundary (p. 19). These issues weaken auditability and calibration but do not invalidate the central controlled contrast or exact algebra.

### D2: domain_accuracy

**score:** `warn`

**evidence:** The estimator formulas, normalization caveats, count-law sufficient statistic, arbitrary-law MaxRL identity, conditional-i.i.d. reduction, RLOO/SFL finite-count relation, and separation of coefficient activity from gradient norm or learning utility are technically careful (pp. 3-5, Section 2; pp. 8 and 12-13). The implementation is tied to stated zero-stabilizer conventions and tested boundary cases. One material scope imprecision remains: the abstract says generically that, “For a coarse unit,” the mean plug-in over-predicts activity by twice the excess all-fail probability (p. 1), whereas the nonnegative sign in Corollary 2 requires the mixture-of-conditionally-i.i.d. atomic-task regime stated only later on p. 5. An arbitrary dependent or under-dispersed count law need not have that sign. The deployment imperative to “score with u_M for M >= your deployed N” (p. 8, Section 3.4 takeaway) is also stronger than the evidence from a bounded exponent sweep supports. These are correctable domain-scope problems rather than failures of the central formulas.

### D3: argumentative_coherence

**score:** `warn`

**evidence:** The core chain is coherent: the same-mean/different-count-law counterexample motivates the sufficient statistic; Proposition 1 and Corollaries 1-2 establish the algebra; MAZE-SCORE identifies the coarse-unit failure; and P0 prospectively changes only the plug-in versus count-law score on that substrate. The paper repeatedly says that the correction does not prove mediation, predict the learning sign, beat p(1-p), or transfer beyond the substrate (pp. 7 and 9). Negative and inconclusive results are retained in the appendix rather than used to rescue claims (pp. 14-19). The warn arises from two local claim-tier inconsistencies: Acrobot is called “Registered and confirmed” in the contribution list (p. 2) despite the paper later conceding incomplete registration timing (p. 9) and only local-lock provenance (p. 17); and the general deployment instruction on p. 8 overextends a one-family, shared-seed exponent sweep whose own discussion says the optimum was not identified (p. 6). The narrower thesis survives, but those statements should be reconciled.

### D4: cross_disciplinary_relevance

**score:** `pass`

**evidence:** The opening counterexample makes the unit-of-analysis problem accessible without specialized estimator knowledge (pp. 1-2, Figure 1). Definitions distinguish task, group, count law, coefficient map, curriculum unit, and learning endpoint; each main formal result receives an interpretation (pp. 3-5). The paper explains paired blocks, treatment delivery, post-hoc diagnosis, SESOI, and the difference between a coefficient envelope and learning utility, and its limitations explicitly name binary rewards, dependence, parameter sharing, optimizer state, scale, and external-validity boundaries (p. 9). Apart from the D2 scope sentence, adjacent-field readers can recover the intended estimands and limitations.

### D5: writing_and_structure

**score:** `pass`

**evidence:** The nine-page main text follows a clear counterexample -> theory -> scoped evidence -> limitations sequence, with the conclusion ending on p. 9 and detailed protocols and negative records routed to pp. 12-19. Tables and captions generally identify the independent unit, interval type, positive-pair count, and status; Figure 2 explicitly warns that some rows use different endpoints and scales (p. 3). Confirmatory, descriptive, development, and inconclusive records are usually labeled at the point of use. A few local presentation issues remain—especially the binned correlation in Figure 5 being juxtaposed with 288,000 contributing group draws (p. 15), comparison of p-value magnitudes across same-seed platform checks (p. 6), and inconsistent spacing/italicization around some p-values—but they do not prevent methodological audit.

## Failure Condition Checks

- **F1 — fired: false.** No mandatory dimension scores `block`; D1-D3 score `warn`.
- **F2 — fired: true.** Three mandatory dimensions—D1, D2, and D3—score `warn`, satisfying the condition that two or more mandatory dimensions score `warn` or worse. The prescribed action is `editorial_decision=major_revision`.
- **F3 — fired: false.** The sole high-priority dimension, D4, scores `pass`, not `block`.
- **F0 — fired: false.** Not every mandatory dimension scores `pass`.

## Review Body

### Methodology Review Report (Peer Reviewer 1)

#### Manuscript Information

- **Title:** Score the Count Law, Not the Mean Pass Rate: Estimator Activity for Curriculum Selection in Verifiable-Reward RL
- **Target:** ICLR 2027
- **Review date:** 2026-08-26
- **Review round:** Round 5

#### Reviewer Identity

Quantitative machine-learning methodology researcher specializing in paired-seed designs, randomization inference, bootstrap uncertainty, preregistration, and reproducible computational experiments.

#### Review Focus

This review evaluates whether the theoretical and computational designs identify the quantities claimed, whether the independent unit and uncertainty procedures match the data-generating process, whether confirmatory status is justified, and whether the central results can be audited and reproduced. Literature completeness and broad field impact are outside scope except where they affect methodological definitions or comparability.

### Overall Assessment

#### Recommendation

- [ ] Accept
- [ ] Minor Revision
- [x] **Major Revision**
- [ ] Reject

This recommendation is contract-derived from F2: D1-D3 each receive `warn`.

#### Confidence Score

**5/5.** The paired-block designs, randomization/sign-flip inference, bootstrap reporting, preregistration, and computational provenance are directly within my expertise.

#### Summary Assessment

The paper develops coefficient activity as a functional of the binary success-count law and derives exact MaxRL, RLOO, and GRPO mass formulas, then tests curriculum consequences through fixed-pool, boundary, and coarse-unit experiments. Its strongest empirical component is a preregistered 48-block paired intervention that changes only the mean-pass-rate plug-in versus count-law score and reports a +.00666 cov-AUC difference with a paired 95% interval and treatment-delivery check. Methodologically, the paper is substantially stronger than a typical multi-seed RL study: it names the independent unit, reports negative and inconclusive branches, separates post-hoc diagnosis from prospective testing, and refuses to equate activity with utility. The main weaknesses are not a failed central design but uneven evidentiary provenance and reporting. Acrobot is called registered and confirmed despite incomplete trusted timing and no prospective power calculation; the “exact” sign-flip interpretation needs an explicit exchangeability/randomization justification; the central P0 raw campaign is external and incompletely disclosed in Appendix D; and the abstract omits the mixture condition required for the nonnegative aggregation-gap statement. These issues are remediable through evidence-tier wording, inferential sensitivity reporting, and an anonymous raw-data/provenance release, but they affect multiple mandatory dimensions and therefore warrant major revision under the contract.

### Strengths

#### S1: Exact unit-of-analysis theory with visible assumptions

Section 2 distinguishes an arbitrary joint binary group law from the conditional-i.i.d. Bernoulli slice and states the finite-stabilizer limitation before presenting the formulas (pp. 3-4). Proposition 1 gives the arbitrary-law identity, while Corollary 2 separately states the mixture-of-conditionally-i.i.d. atomic-task assumption (p. 5). Appendix A provides derivations, boundary cases, and a dependent-rollout stress test (pp. 12-13). This is a strong methodological match between theorem scope and the data-generating unit.

#### S2: Strong prospective isolation in the P0 intervention

The P0 comparison holds estimator, N, warm start, task generator, budget, posterior state/update, decay, floor, and 48 paired seeds fixed, varying only the score functional (p. 7, Section 3.4). Its frozen protocol identifies the seed/warm-start block as the independent unit, counterbalances execution order, requires complete terminal artifacts, predeclares one endpoint and one confirmatory contrast, and uses a delivery gate. That design directly answers the narrow causal question the paper asks on this substrate.

#### S3: Correct treatment of correlated observations and repeated platforms

The paper does not count rollout groups, evaluations, levels, or samplers within a block as independent replications. It explicitly says that the two platform results reuse seeds and are portability checks rather than new independent seed cohorts (p. 6); Figure 6 says one point is one level, not an independent training replicate (p. 15); and the maze factorial does not pool sampler observations as independent (p. 17). This guards against a common and serious ML pseudoreplication error.

#### S4: Negative results and claim boundaries are first-class

MAZE-SCORE's reversal, the AMaze standalone-priority failure, the full-budget gate's inconclusive verdict, the Digits interaction failure, the allocation-mechanism failure, and incomplete Countdown pairing are all retained and labeled at their earned level (pp. 7 and 14-19). The manuscript repeatedly states that activity is not gradient norm, expected improvement, or a learning guarantee (pp. 4, 7, and 9). This transparency materially reduces selective-reporting and HARKing risk.

### Weaknesses

#### W1: Confirmatory provenance is uneven across headline studies

**Problem:** The contribution list calls Acrobot “Registered and confirmed” (p. 2), but the manuscript later states that registration timing remains incomplete (p. 9). Appendix C reports that the source lock is local and that independent pre-execution timing would require a trusted copy or repository history (p. 17). The Acrobot result record also acknowledges that 20 seeds were inherited rather than selected by prospective power analysis.

**Why it matters:** “Registered” and “confirmed” imply prospective, independently auditable constraints on researcher degrees of freedom. A local hash demonstrates internal consistency but not timing, and the absence of prospective power weakens the interpretation of both success and potential non-significance. Because Acrobot is the main fixed-pool positive, tier inflation affects the contribution perimeter.

**Suggestion:** Supply an anonymous trusted timestamp or immutable pre-execution commit that binds the V2 protocol, lock, analyzer, and seed list. If none exists, change the main-text status to “internally frozen, controlled result” and reserve “preregistered/confirmed” for studies with independently verifiable timing. Add the prospective sensitivity/MDE implied by n=20 or explicitly state that no prospective power claim is available.

**Severity:** Major.

#### W2: “Exact sign-flip” inference needs its exchangeability basis or a robustness alternative

**Problem:** The primary analyses use exact two-sided paired sign-flip p-values (pp. 6-7), while Appendix C correctly concedes for Acrobot that pairing and common random numbers do not guarantee sign exchangeability (p. 17). P0 counterbalances process order and runs both arms per block, but neither that counterbalancing nor deterministic shared-seed pairing is itself a randomized treatment-label assignment. The main paper does not state the symmetry/exchangeability model under which the sign-flip p-values are exact.

**Why it matters:** Calling a p-value “exact” can be read as design-based randomization inference. Without randomized arm labels or a justified symmetric paired-difference null, the enumeration is exact computationally but its null calibration is assumption-dependent. P0's 40/48 signs and interval make the substantive result unlikely to hinge on this, but the inferential claim should be technically precise.

**Suggestion:** State whether arm labels/order were randomized and what sharp-null exchangeability follows. Otherwise call the procedure an exact enumeration under a sign-exchangeability assumption, report a two-sided binomial sign test based only on 40/48 positive blocks as a distribution-light sensitivity, and add a studentized or wild-bootstrap sensitivity that does not rely on the raw-mean sign-flip symmetry. Report paired SD or IQR/range alongside each primary so readers can inspect skew and outliers.

**Severity:** Major.

#### W3: The central P0 result is not reproducible from raw outcomes in the supplied artifact

**Problem:** The repository contains the frozen P0 analysis JSON and tests of the analyzer, but its 96 run inputs point to an external absolute `/data/...` campaign. The compact registry acknowledges that the validated raw P0 campaign remains external, whereas Appendix D's enumerated missing-data paragraph lists maze checkpoints, Countdown records, paid-probe raw, and Digits replay data but does not explicitly list P0 raw (p. 19). The deposit inventory is not yet deposited, and its described payload does not include P0 raw data.

**Why it matters:** P0 is the paper's causal closure. Checking a frozen summary hash and regenerating a figure is not the same as independently re-running the locked analyzer from the complete 2x48 endpoint and telemetry records. The current reproducibility statement can therefore be read more broadly than the actual local path supports.

**Suggestion:** Include, in an anonymous supplement or archive, the 96 endpoint JSONLs, 24,000 telemetry rows, per-block manifests/receipts, frozen analyzer, environment/source manifests, and a relative-path reanalysis entrypoint. If release is impossible, add P0 explicitly to Appendix D's external-boundary list and say that `reproduce.sh` verifies the stored summary but cannot reconstruct the primary from raw runs.

**Severity:** Major.

#### W4: Two scope statements exceed the assumptions or empirical perimeter

**Problem:** The abstract's coarse-unit sentence omits the mixture-of-conditionally-i.i.d. atomic-task condition needed for the nonnegative gap (p. 1), even though Corollary 2 states it correctly on p. 5. The deployment takeaway then instructs readers to choose a score exponent M at least as large as deployed N (p. 8), although the exponent evidence comes from a bounded, co-designed sweep and shared-seed portability checks, with u64—not deployed u16—best only at the tested operating point (p. 6).

**Why it matters:** The first sentence can be false for an arbitrary dependent or under-dispersed count law; the second converts a scoped empirical boundary into a general prescription. Both risk reintroducing exactly the mean/count-law and activity/utility over-inferences the paper otherwise handles well.

**Suggestion:** Add “under a mixture of conditionally i.i.d. atomic tasks” to the abstract statement. Replace the deployment imperative with: “In our tested fixed-pool sweep, harder-peaked exponents beyond deployed N continued to improve performance; deployed N was not an identified optimum.” Keep any broader M >= N rule as a hypothesis for future validation.

**Severity:** Major.

### Detailed Comments

#### Research Questions and Hypotheses

- The central question is clear and answerable: when is mean pass rate sufficient for estimator activity, and does correcting a coarse-unit plug-in improve a downstream endpoint? Figure 1 and the first paragraph frame the estimand rather than only a method name (pp. 1-2).
- The paper appropriately separates the proved activity identity from the empirical utility consequence. The P0 hypothesis tests only causal relevance on one substrate; it does not test mediation, universal transfer, or superiority to p(1-p) (p. 7).
- The fixed-pool score-shape question is less cleanly tiered because Acrobot's prospective timing cannot be independently verified from the supplied record. That status should be corrected, not used to discard the observed controlled contrast.

#### Research Design

- P0 is a strong paired-block controlled experiment. The only intended treatment is the score functional applied to a common posterior definition, and induced divergence in later visits/policies is part of the treatment strategy rather than a confound.
- MAZE-SCORE is correctly used as a registered boundary and as a source of post-hoc calibration diagnosis, not as proof that calibration mediated its negative learning endpoint (p. 7).
- The Acrobot portability runs share seeds and a deterministic engine. Calling them portability checks, not two independent replications, is correct (p. 6).
- AMaze replacement is explicitly a five-seed development negative at one-sixth budget; the paper does not use it as confirmatory baseline superiority (pp. 3 and 7). The ten-pair full-budget gate is appropriately reported as inconclusive (p. 14).

#### Sampling Strategy

- Independent units are correctly identified as paired seed/warm-start blocks. P0's n=48 is justified prospectively for 90.1% simulated support probability at a powered-for +.0075 effect under the pessimistic historical SD, though the power memo also shows only about 50.3% support probability at the +.005 SESOI. The manuscript should summarize both facts, not merely the observed success.
- Acrobot n=20 was inherited without prospective power. The two later platforms reuse those seeds, so they do not increase the inferential sample size.
- Secondary studies with n=3, n=5, n=6, n=8, or n=10 are generally labeled development, descriptive, or inconclusive. This is appropriate and should be preserved.

#### Data Collection

- Binary outcomes are verified, budgets are named in environment transitions or completed groups/updates, evaluation grids are fixed, and complete terminal state is required. P0's delivery measure (mean visit TV .33597 against a .05 threshold) demonstrates that the two policies actually received meaningfully different curricula (p. 7).
- The paper distinguishes aggregate evaluation records from raw per-task outcomes and correctly refuses to call the Countdown best@16 proxy standard pass@16 (p. 18).
- The artifact would be stronger if every central result retained raw per-task or per-evaluation outcomes. The P0 raw campaign, maze checkpoints, Countdown outcomes, and several large replay payloads are not local.

#### Analysis Methods

- Raw paired mean differences in normalized/cov-AUC are appropriate effect measures for the declared endpoints; SESOIs provide practical interpretation. Bootstrap intervals resample at the paired block level, which matches the independent unit.
- Holm correction is used for Acrobot's two uniform secondaries and MAZE-SCORE's two comparisons. P0 has one primary, so no within-study multiplicity correction is needed. The limitation that separate alpha=.05 primaries do not define a paper-wide error rate is appropriately explicit (p. 9).
- The bootstrap specification is mostly reproducible (percentile interval, resample count, fixed seed). The paper should also state whether intervals are percentile or BCa in the main/appendix statistical-method paragraph and report paired dispersion.
- Sign-flip calibration is the main analysis concern; see W2. A counterbalanced execution order controls order effects but does not automatically yield arm-label randomization.
- No parametric t-test assumption diagnostics are needed for the primary exact/sign-flip procedures, but bootstrap i.i.d.-block assumptions and seed-population generalization should be stated. The paid-probe t-test on p. 17 should include a paired-difference normality/robustness diagnostic, especially because p=.05149 and its percentile interval excludes zero.

#### Results Presentation

- Table 1 reports arm means, paired differences, 95% intervals, exact p-values, multiplicity, and decisions (p. 6). P0 additionally reports treatment delivery and positive blocks (p. 7). This is substantially more informative than p-values alone.
- Figure 2 usefully distinguishes frozen primaries from development rows and notes that heterogeneous endpoints share a panel only for direction (p. 3). A faceted or normalized display would nevertheless reduce the risk of visual magnitude comparison across incompatible scales.
- Figure 5's r=.90 is a correlation across bins, while 288,000 is the number of contributing group draws (p. 15). Report the number of bins as the correlation sample size and retain 288,000 only as the underlying draw count.
- The sentence emphasizing that the same-seed platform checks have smaller p-values than V2 (p. 6) should be removed. Cross-study p-value magnitude is not an effect-size or replication-strength comparison, especially with reused seeds.

#### Discussion and Conclusion

- The limitations section is admirably direct about activity not being a learning guarantee, platform sensitivity, small models, one maze family, no paper-wide error rate, post-hoc diagnosis, and incomplete raw-data/registration provenance (p. 9).
- The conclusion stays within the count-law and one-substrate intervention story. The p. 8 operational rule should be narrowed as in W4 so it matches those limitations.

#### Reproducibility

- Strengths include exact-enumeration tests, implementation-parity checks, frozen analyzer hashes, a 96/96 quantitative claim trace, a compact manifest covering every included figure, and a one-command portable check (pp. 9 and 19).
- The portable command verifies stored inputs and regenerates figures, but it does not reconstruct every headline endpoint from raw execution records. That distinction should be explicit in the reproducibility statement.
- P0's git history separates the 2026-08-20 freeze from the 2026-08-26 result commit, which is good repository evidence. The anonymous release should bind those commits and the external campaign receipts in a trusted archive.

#### Methodological Fallacies and Red-Flag Scan

- **Pseudoreplication:** Not detected in the primary analyses. The paper explicitly prevents levels, samplers, evaluation checkpoints, and shared-seed platform runs from being counted as independent seeds.
- **Selective reporting/confirmation bias:** No strong evidence. Negative, failed-delivery, and inconclusive branches are visible and often receive stricter language than positive studies.
- **HARKing:** The MAZE-SCORE calibration diagnosis is explicitly post-hoc; P0 is a subsequent prospective intervention. The Acrobot registration label remains a medium provenance concern because trusted timing is incomplete.
- **Multiplicity:** Within-study families are generally controlled or labeled descriptive. The absence of a paper-wide error rate is disclosed; no universal omnibus claim should be inferred from the collection of separate primaries.
- **Causal overreach:** P0 supports a controlled score-policy contrast on one substrate. The paper correctly declines mediation and transfer claims, but the deployment imperative should be narrowed.
- **Assumption risk:** Exact sign-flip p-values require sign exchangeability or a randomized-label design; that basis is not fully established in the manuscript.

### Statistical Reporting Completeness

**Overall level:** **Adequate (77/100).** The field-appropriate raw effect differences, paired 95% intervals, exact p-values, independent n, positive-pair counts, multiplicity rules, and SESOIs are strong. The main deficits are incomplete paired descriptives, incomplete prospective power coverage outside P0, insufficient articulation of sign-exchangeability/bootstrap assumptions, and incomplete raw-data availability.

| Component | Score | Methodology assessment |
|---|---:|---|
| Descriptive statistics | 10/15 | Arm means and independent n are reported for central studies; paired SD, range/IQR, and distribution plots are often absent. |
| Effect-size reporting | 18/20 | Raw AUC/cov-AUC differences and SESOIs are directly interpretable; standardized paired effects are not essential but paired dispersion should accompany them. |
| Confidence intervals | 15/15 | Central effects carry paired 95% intervals with named bootstrap procedures in bound records. |
| Assumption reporting | 8/15 | Independent units are explicit, but sign exchangeability and bootstrap block-population assumptions need fuller treatment. |
| Statistical power | 7/10 | P0 has prospective full-rule calibration; Acrobot explicitly lacks prospective power and several small-n results are only bounded by status labels. |
| Missing-data handling | 8/10 | Central campaigns fail closed and do not substitute seeds; some raw records are external, and P0 is not fully itemized in Appendix D. |
| Statistical format | 8/10 | Reporting is generally clear; a few p-value spacing/italicization and correlation-n labels need correction. |
| Red-flag control | 3/5 | Negative-result transparency is excellent; registration timing and sign-flip calibration remain medium concerns. |

**Specific statistical recommendations:**

1. Add one compact statistical-method paragraph defining the independent unit, bootstrap type/resamples, sign-flip null assumption, multiplicity family, SESOI, and missing-block rule for every primary.
2. Report the paired SD and range or IQR of block-level differences for Acrobot, MAZE-SCORE, and P0; provide block-level dot/interval plots in the appendix.
3. Summarize P0's prospective power calibration, including that it was powered for +.0075 rather than the +.005 SESOI; give an MDE/sensitivity statement for Acrobot's inherited n=20.
4. Add a sign-test and studentized/wild-bootstrap robustness analysis for each load-bearing paired primary, or explicitly justify design-based arm-label exchangeability.
5. Clearly distinguish group-draw counts from the number of independent bins/blocks used in correlations and intervals.

### Questions for Authors

1. What independently verifiable pre-execution object binds the Acrobot V2 protocol, source/runtime lock, analyzer, and seed list? If none exists, will you remove “registered and confirmed” from the contribution perimeter?
2. What design feature or stochastic model establishes sign exchangeability of the paired block differences for Acrobot, MAZE-SCORE, and P0? Can you report a sign-test and a studentized or wild-bootstrap sensitivity alongside the enumerated sign-flip result?
3. Will the anonymous submission artifact include P0's complete 2x48 endpoint/telemetry campaign and a relative-path invocation of the frozen analyzer? If not, why is P0 absent from Appendix D's explicit list of external raw-data boundaries?
4. Will you qualify the abstract's aggregation-gap sentence with the mixture-of-conditionally-i.i.d. atomic-task assumption and narrow the M >= N deployment instruction to the tested exponent sweep?

### Minor Issues

- On p. 6, remove “both with a smaller p-value than V2's own”; compare platform effects and intervals, not p-value magnitude.
- On p. 7, standardize mathematical p-value typography and spacing; similar compact forms such as `p=.0625` occur elsewhere.
- In Figure 5 (p. 15), label the number of bins used for r=.90 separately from the 288,000 underlying group draws.
- In Figure 2 (p. 3), consider faceting rows by endpoint or suppressing a common quantitative axis so readers do not compare magnitudes across incompatible scales.
- In Appendix D (p. 19), distinguish “stored-summary reproduction,” “raw reanalysis,” and “full training reproduction” rather than using one broad reproducibility label.

### Ordinal 0-100 Rubric Scores

These scores are ordinal quality indicators, not cardinal acceptance probabilities. The contract's failure-condition action, reported below, takes precedence over the weighted rubric mapping.

| Dimension | Score | Descriptor | Methodology note |
|---|---:|---|---|
| Originality (20%) | 83 | Strong | A novel count-law formulation and exact estimator-specific decomposition, assessed here only for methodological distinctness. |
| Methodological Rigor (25%) | 73 | Adequate | Strong P0 and exact theory, offset by provenance, inferential-assumption, power, and raw-reproduction gaps. |
| Evidence Sufficiency (25%) | 72 | Adequate | Main claims have proofs or paired studies, but external validity and several raw-data boundaries remain narrow. |
| Argument Coherence (15%) | 76 | Strong | The central chain is coherent; registration-tier and deployment-language inconsistencies are localized but material. |
| Writing Quality (15%) | 82 | Strong | Clear structure, strong captions and limitations, with a few scope and statistical-label issues. |
| **Weighted average** | **76.6** | **Minor Revision by the generic ordinal mapping** | The binding sprint contract instead requires **Major Revision** because F2 fired. |

## Editorial Decision

**Major Revision.** F2 is the highest-severity fired condition: D1, D2, and D3 are mandatory dimensions and each scores `warn`. No `block` condition fires, so rejection is not indicated. Revision should focus on (1) independently auditable confirmatory-status wording for Acrobot, (2) an explicit inferential basis and robustness checks for sign-flip tests, (3) raw or precisely disclosed P0 reproducibility boundaries, and (4) assumption-accurate abstract and deployment language.
