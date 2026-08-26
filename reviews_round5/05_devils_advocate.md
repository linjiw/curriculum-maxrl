## Dimension Scores

| Dimension | Contract signal | Basis under the Phase 1 trigger |
|---|---|---|
| D1: Methodology Rigor | `warn` | The prospective P0 comparison credibly isolates a change of scoring functional on one substrate, so the narrower algorithmic contrast remains interpretable. However, it does not identify activity calibration as the mechanism, and the manuscript leaves the audit status of central “preregistered/confirmed” records ambiguous while acknowledging missing timing and raw-data material. These are serious qualifications, but they do not require replacing the core design. |
| D2: Domain Accuracy | `warn` | The count-law mass identity appears internally consistent and is carefully distinguished from learning utility. The warning is triggered because the “exact bridge” to expected update is stated without the conditional-independence/conditional-score assumptions needed to collapse score geometry to two marginal means, and because the recommendation to use `u_M` for `M>N` is no longer estimator-matched activity for the deployed estimator. |
| D3: Argumentative Coherence | `warn` | The exact diagnostic claim survives, but the prescriptive inference does not follow cleanly. The paper rejects deployed-`N` peak specificity, reports weak or reversed activity–benefit localization, and shows learning improvements accompanied by lower realized mass, yet still recommends activity-shaped scoring. This is a logic-chain failure for the deployment prescription, not a refutation of the count-law theorem or the scoped P0 treatment contrast. |
| D4: Cross-Disciplinary Relevance | `warn` | A genuine bridge exists among estimator algebra, count distributions, and curriculum scoring. Transfer guidance remains underdetermined for practical RLVR, however: the central neural evidence is one small maze substrate, fixed-`N` assumptions are prominent, the faithful PLR mapping is explicitly outside the evidence, and estimation/nonstationarity costs are not developed into operational diagnostics. |
| D5: Writing and Structure | `warn` | The paper is unusually explicit about negative results and evidence boundaries, and its nine-page argument is traceable. Nevertheless, the imperative title, the broad first sentence of the conclusion, and shifting labels such as “registered,” “internally frozen,” and incompletely auditable records invite stronger readings than the detailed caveats permit. |

### Failure Condition Checks

- **F1 — any mandatory `block`: not fired.** Inputs are D1=`warn`, D2=`warn`, and D3=`warn`; none is `block`.
- **F2 — two or more mandatory dimensions `warn` or worse: reviewer-local predicate fired.** All three mandatory dimensions are `warn`. This report therefore contributes one positive F2 reviewer input. The panel condition for `major_revision` is not decidable here: it fires only if at least 4 of 5 reviewers satisfy their local predicate.
- **F3 — any D4 `block`: not fired.** D4=`warn`.
- **F0 — accept only if all mandatory dimensions pass and no higher rule fires: not satisfied.** None of D1–D3 is `pass`, and the reviewer-local F2 predicate fired.

## Review Body

### Devil’s Advocate Review

The manuscript’s strongest feature is its refusal to equate coefficient mass with gradient quality or learning progress. It also exposes negative and inconclusive studies more candidly than is typical, which makes the remaining inference gap unusually easy to locate.

### Strongest Counter-Argument

The strongest rival account is that the successful curricula work by reshaping difficulty and visitation, not by matching the deployed estimator’s coefficient activity. The manuscript’s own results fit this account at least as well as the proposed activity narrative. With the estimator fixed at `N=16`, `u64` trends above `u16`, so the preferred score is not the deployed estimator’s activity curve. Exact continuation utility peaks around `p≈.02`, below both activity peaks, which the paper summarizes as activity ranking utility well but locating it poorly. In P0, substituting the count-law score changes the visitation distribution and improves cov-AUC, but the per-level activity-gap/coverage association is only `ρ=.157` and reverses at level 5. More damagingly, the allocation studies improve pass@8 while reducing realized coefficient mass in essentially every comparison. Digits likewise shows that estimator-shaped samplers can both lose to uniform.

This rival explanation does not refute Proposition 1, the same-mean counterexample, or the fact that P0’s scoring-rule substitution caused an endpoint difference on one maze substrate. It does refute the stronger inference that closer coefficient-activity calibration is why the curriculum improves, or that estimator matching supplies a generally privileged selection rule. P0 jointly changes hardness, visitation entropy, floor usage, and exposure across levels; calling the treatment a count-law correction does not isolate which induced distributional change matters. The durable contribution is therefore an exact estimator diagnostic and a warning about coarse pooling. Turning that into a curriculum prescription requires either a separately tested mediation claim or a narrower statement that activity generates hypotheses whose utility must be independently validated.

### Issue List

#### CRITICAL

No CRITICAL issue is warranted. The exact count-law coefficient-mass claim retains value even if every curriculum inference below is narrowed. The vulnerabilities concern the bridge from that theorem to selection utility, so they meet the contract’s `warn` rather than `block` triggers.

#### MAJOR

##### M1. The deployed-`N` activity rationale is abandoned by the paper’s own recommended score

- **severity:** MAJOR
- **evidence:** Proposition 1 and Corollary 1 define `u_N` as the half-mass of the estimator deployed with group size `N` (PDF pp. 4–5, §2). The exponent experiment fixes the estimator at `N=16`, finds `u64` at the sweep maximum, and rejects deployed-`N` peak specificity (PDF p. 6, §3.2). The operational takeaway nevertheless says to “score with `u_M` for `M ≥` your deployed `N`” and calls deployed `N` a floor (PDF p. 8, §3.4, Takeaway).
- **reasoning:** Once `M≠N`, `u_M` is not the coefficient activity of the deployed estimator. It is a harder-peaked member of the same parametric family. The result therefore supports hardness shaping or a tunable curriculum temperature, not the claim that the estimator determines the correct curriculum geometry. This is a genuine logic-chain break for the deployment prescription. It is not merely narrowness and does not invalidate the exact `u_N` identity.
- **actionable_fix:** Separate “estimator-matched activity” (`M=N`) from an empirical hardness/exponent hyperparameter (`M`). Remove the claim that deployed `N` is a general floor unless a prospective multi-domain test supports it. Present the sweep as evidence against activity-peak matching, and retain P0—not the exponent sweep—as the scoped evidence for scoring the actual count law.
- **confidence:** High.

##### M2. P0 identifies a scoring-rule effect, not the claimed activity-correction mechanism

- **severity:** MAJOR
- **evidence:** P0 varies the plug-in versus count-law scoring functional and reports `+.00666` cov-AUC (PDF pp. 7–8, §3.4). The manuscript properly says this does not prove endpoint mediation (PDF p. 8). Its registered per-level secondary finds only `ρ=.157`, includes a level with a large activity gap and negative coverage contrast, and states that the pattern is not mediation evidence (PDF p. 15, Fig. 6 and Appendix C). The allocation factorial improves pass@8 while realized coefficient mass falls in all 16 primary comparisons; the expanded follow-up has positive learning contrasts while mass is lower in 28/32 cells (PDF pp. 17–18, Appendix C).
- **reasoning:** The intervention causally identifies the package “sample according to functional A rather than B.” It does not identify improved activity calibration as the operative cause, because the scoring change necessarily changes difficulty, visitation, entropy, exposure, and training-state trajectories. The paper’s contrary mechanism records make the simpler visitation/hardness account at least as plausible. The text often states the limitation, but phrases such as “confirming causal relevance of the correction” and the imperative title let the algorithmic contrast borrow mechanistic meaning it has not earned.
- **actionable_fix:** Replace “causal relevance of the correction” with the precise estimand: “substituting the count-law scoring rule improved cov-AUC on this substrate.” State that activity mediation is unsupported. If mechanism is needed for the contribution, preregister an intervention or analysis that varies activity calibration while matching induced difficulty/visitation statistics, or demonstrate that changes in delivered mass predict independent seed-block endpoint changes under a frozen rule.
- **confidence:** High.

##### M3. The exact expected-update bridge is missing assumptions precisely where the count-law generalization matters

- **severity:** MAJOR
- **evidence:** The setup explicitly permits an arbitrary joint binary group law `Q_x` (PDF p. 3, §2), and Proposition 1 emphasizes that no independence or identical distribution is required (PDF p. 4). Immediately afterward, the paper says the bridge is exact, `E[ĝ|x]=u_N(p)(μ_+−μ_−)`, with `μ_±=E[s_i|r_i=1/0,x]`, without restating a conditional-i.i.d. assumption (PDF p. 4). The introduction lists this bridge among the reasons the coefficient surrogate is defensible (PDF p. 2).
- **reasoning:** Under a general dependent group law, the score distribution for member `i` can depend on the total count `K` even after conditioning on `r_i`. Expected updates then involve count-conditioned quantities such as `E[s_i|r_i,K,x]`, and cannot in general be reduced to two marginal conditional means times `u_N(p)`. The mass identity is arbitrary-law exact; the displayed update factorization is not arbitrary-law exact without extra conditional-independence/exchangeability assumptions. Because the paper’s novelty is precisely to leave the atomic i.i.d. slice, this unstated boundary weakens the principal bridge from algebra to curriculum relevance.
- **actionable_fix:** State and prove the bridge as a separate atomic conditional-i.i.d. proposition, including the assumption that a member’s score is conditionally independent of peer outcomes given its own response/outcome. Give the arbitrary-count-law expression with `K`-conditioned score means, and explicitly say that P0’s coarse-unit activity correction does not inherit the scalar factorization.
- **confidence:** High.

##### M4. The evidentiary labels are stronger and more uniform than the disclosed audit record

- **severity:** MAJOR
- **evidence:** The abstract calls P0 “preregistered,” and the introduction labels Acrobot “Registered and confirmed” and P0 “Preregistered correction, confirmed” (PDF pp. 1–2). The Acrobot methods instead call the primary “internally frozen” (PDF p. 6, §3.1). The limitations say registration timing and raw-data availability remain incomplete for “Digits, Acrobot and the maze records” (PDF p. 9, §5). Appendix D says the compact registry contains only one P0 analysis artifact, lists maze checkpoints among missing external data, and distinguishes locally hashed locks from public pre-execution timing evidence in several secondary records (PDF pp. 16–19).
- **reasoning:** “Preregistered,” “internally frozen,” “source/runtime locked,” and “prospective” are not interchangeable evidentiary states. From the PDF alone, it is unclear which timing caveat applies to P0 and which underlying records permit independent reconstruction of its load-bearing endpoint. That ambiguity matters because prospectivity is central to distinguishing the successful correction from the post-hoc granularity diagnosis.
- **actionable_fix:** Add a compact per-study provenance table naming the freeze time, immutable public identifier or commit, first-run time, analyzer hash, independent replicate, raw-data availability, and exact evidence label. Explicitly exempt P0 from the blanket “maze records” caveat if warranted; otherwise replace “preregistered/confirmed” with a label supported by the disclosed record and explain the limitation in the abstract or main evidence paragraph.
- **confidence:** Medium, because the phrase “maze records” may not be intended to include every P0 timing object, but the manuscript does not resolve that ambiguity.

##### M5. The conclusion converts a scoped Acrobot result into a category-level statement contradicted by another own-unit experiment

- **severity:** MAJOR
- **evidence:** The conclusion begins, “Where the curriculum scores the estimator’s own unit, the deployed-`N` shape beat both `u2` and uniform sampling” (PDF p. 9, §6). Yet the exact-probability Digits factorial is also described as an own-unit boundary: MaxRL’s `u8` beats `p(1−p)` but both matched samplers are below uniform, with `u8−uniform = −.11279` (PDF p. 16, Appendix C). The main evidence section already acknowledges that both Digits matched samplers fall below uniform (PDF p. 6, opening of §3).
- **reasoning:** The conclusion’s condition (“scores the estimator’s own unit”) is not sufficient for the claimed outcome (“beat both `u2` and uniform”) under the paper’s own evidence. This is a direct data–conclusion mismatch, though it is repairable because the Acrobot result itself remains intact. It also encourages readers to mistake a necessary granularity condition for a sufficient curriculum-success condition.
- **actionable_fix:** Rewrite the sentence as a named, substrate-specific result: “In the Acrobot fixed pool, where the scored and estimator units coincide, `u16` beat `u2` and uniform.” Immediately note that Digits satisfies the unit condition but loses to uniform, proving that correct granularity does not imply curriculum utility.
- **confidence:** High.

##### M6. The paper’s RLVR-facing prescription outruns its demonstrated transfer bridge

- **severity:** MAJOR
- **evidence:** The direct confirmed fixed-pool result is a 640-parameter Acrobot policy; the neural P0 result is one 1.26M-parameter transformer and procedural-maze family (PDF pp. 6–8, §§3.1–3.4). The AMaze student does not form binary rollout groups, and the paper says the faithful mapping is outside the evidence (PDF p. 7, §3.3). The limitations state that the maze result covers one family, architecture, `N`, and budget and that provenance gaps preclude a broad neural claim (PDF p. 9, §5). Nevertheless, the title names verifiable-reward RL generally and the takeaway opens “If you are deploying this” (PDF p. 8).
- **reasoning:** This is not a refutation of the theorem; it is a cross-disciplinary transfer gap. Practical LLM RLVR often includes changing policies, heterogeneous prompt templates, variable completion counts, length/token weighting, and estimator details beyond binary group coefficients. The manuscript has not yet shown which of those preserve the operational meaning of its score or when count-law estimation is stable enough to guide sampling.
- **actionable_fix:** Scope the title and deployment paragraph to fixed-`N`, binary, finite-group estimators and the tested substrates. Add a short transfer checklist identifying the consumed unit, group law, coefficient map, nonstationarity window, required count bins, and scoring cost. Keep LLM-scale use explicitly as a hypothesis until a prospective test is available.
- **confidence:** High.

#### MINOR

##### m1. “Comes free” hides the estimation problem created by replacing a mean with a count law

- **severity:** MINOR
- **evidence:** The takeaway says both the plug-in and count-law quantities “come free from group outcomes you already log” (PDF p. 8, §3.4). Algorithm 1 requires decayed sufficient statistics or a full count-bin posterior, a floor, concentration, and an i.i.d.-Binomial prior (PDF p. 5, Algorithm 1). The paper separately notes that posterior choice, decay, floor, and concentration remain empirical choices (PDF p. 5).
- **reasoning:** No additional rollout may be required, but estimation is not free: the count law has higher state and sample demands, is policy-nonstationary, and may be sparse for many coarse units. Those costs can change the comparison against a mean-based score and are central for readers deciding whether the count-law correction is practical.
- **actionable_fix:** Replace “come free” with “require no additional rollouts when group counts are retained.” Report posterior sensitivity or effective observations per unit in P0, and state when the sufficient-statistic shortcut is available versus when a full count histogram is required.
- **confidence:** High.

##### m2. The title suppresses the atomic regime in which mean pass rate is sufficient

- **severity:** MINOR
- **evidence:** The title says “Score the Count Law, Not the Mean Pass Rate.” The abstract and Corollary 1 establish that under the atomic conditionally-i.i.d. model, the count law collapses to a function of `p`, and the paper successfully uses that scalar score in Acrobot (PDF pp. 1 and 5).
- **reasoning:** The imperative title reads as a universal rejection of mean-pass-rate scoring, while the actual result is conditional: score the count law when the curriculum unit is coarser than the estimator’s atomic unit or when dependence invalidates the scalar reduction. The body corrects the impression, but the title is the paper’s most visible claim.
- **actionable_fix:** Qualify the title around granularity, for example by naming coarse curriculum units or the insufficiency of mean pass rate outside the atomic i.i.d. model.
- **confidence:** High.

### Ignored Alternative Explanations/Paths

These are rival accounts rather than additional issues; their evidence and remedies are recorded in M1–M2.

1. **Hardness-temperature account:** `u_M` succeeds as a tunable left-shifted difficulty curve, independently of whether `M` matches the estimator group size.
2. **Visitation-regularization account:** the count-law arm helps by avoiding heterogeneous levels that produce net-negative coverage learning, not because higher realized coefficient mass mediates improvement.
3. **Learning-progress account:** a curriculum based on change, forgetting, critic disagreement, or directly estimated continuation utility could dominate instantaneous coefficient mass while preserving the count-law diagnostic as a gate.

### Missing Stakeholder Perspectives

- Practitioners running large-scale RLVR with variable group sizes, completion lengths, and nonstationary prompt pools.
- Maintainers and users of SFL/ProCuRL/PLR-style baselines whose full selection and cost semantics differ from score-only ablations.
- Artifact auditors who must determine which prospective claims can be reconstructed from locally available raw outcomes and timing records.

### Unexamined Premise

##### U1. A larger estimator-controlled contrast envelope is treated as the privileged pre-rollout resource despite evidence that the useful resource may be elsewhere

- **severity:** MAJOR
- **evidence:** The paper motivates `ℓ1` mass as a worst-case update envelope knowable before rollout (PDF p. 4, §2) and turns it into a sampler in Algorithm 1 (PDF p. 5). It simultaneously states that direction, parameter sharing, optimizer state, transfer, and non-i.i.d. rollouts determine downstream effect (PDF pp. 4 and 9), and reports improved learning with lower mass in the allocation studies (PDF pp. 17–18).
- **reasoning:** An upper envelope is valuable as an exact zero diagnostic, but ranking nonzero tasks by that envelope assumes that potential coefficient magnitude is the scarce curriculum resource. The manuscript never establishes that premise, and several records point toward continuation utility, information bandwidth, or learning progress instead. The assumption survives because the paper correctly disclaims a learning law while still using the same quantity as the default selection score.
- **actionable_fix:** Recast activity as a feasibility gate or diagnostic unless and until its ranking value is prospectively established. Compare a two-stage curriculum—activity excludes exact or estimated dead zones, then a separate utility/progress score ranks live units—against pure activity ranking under matched budgets.
- **confidence:** High.

### Observations (Non-Defects)

- The arbitrary-law mass identity and the same-mean/different-count-law construction remain useful even under the strongest rival account.
- The explicit reporting of Digits, AMaze, exponent-sweep, allocation, and provenance boundaries materially reduces the risk that readers mistake every positive pattern for confirmation.

## Editorial Decision

This devil’s-advocate role does **not** issue a balanced accept/reject recommendation or an ordinal manuscript score. Its contract inputs are: D1=`warn`, D2=`warn`, D3=`warn`, D4=`warn`, D5=`warn`; F1 did not fire; the reviewer-local F2 predicate did fire because three mandatory dimensions are warn-or-worse; F3 did not fire; and F0 is not satisfied. Panel synthesis should count this as one positive F2 reviewer input and apply `major_revision` under F2 only if at least 4 of 5 reviewers meet the same local predicate.
