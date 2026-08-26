## Dimension Scores

### D1 — methodology_rigor: `pass`

The paper identifies the relevant observational and intervention units unusually clearly. Proposition 1 is stated for an arbitrary joint binary group law, with conditional i.i.d. used only for the scalar reduction (PDF pp. 3–5); Algorithm 1 exposes the per-unit sufficient statistics and sampling floor (p. 5); and the P0 experiment holds the estimator, group size, warm start, generator, budget, posterior machinery, decay, floor, and paired blocks fixed while varying the score functional (pp. 7–8). The paper also separates independent blocks from correlated evaluations, distinguishes fixed-transition and fixed-completion clocks, and discloses raw-data and registration gaps (pp. 9, 15–19). These choices support the paper's scoped theory and same-substrate causal claim. I do not treat the secondary and appendix experiments as confirmatory evidence because the manuscript itself does not.

### D2 — domain_accuracy: `pass`

The domain distinctions are accurate and materially useful. The manuscript separates coefficient mass from gradient norm, SNR, update direction, and learning utility (pp. 3–4); restricts the nonnegative aggregation gap to the shared-unit, conditionally-i.i.d. mixture setting (p. 5); gives the arbitrary-law identity separately (pp. 4, 13); and recognizes SFL as realized RLOO count-law scoring up to a fixed factor rather than dismissing it as a heuristic (p. 8). It also distinguishes task selection from rollout allocation and from task creation (pp. 8, 13–14). I found no central domain conflation that would warrant `warn` or `block`.

### D3 — argumentative_coherence: `warn`

The core chain—same mean/different count law, exact estimator geometry, bounded fixed-pool evidence, coarse-unit failure, then a score-only correction—is coherent and appropriately cautious (pp. 1–9). The `warn` is for evidentiary status language: the contribution list calls the Acrobot result “Registered and confirmed” (p. 2), while the study is described as internally frozen (p. 6) and the limitations state that registration timing for Acrobot and maze records remains incomplete (p. 9). The appendix likewise distinguishes local mechanical provenance from independently time-established public registration for several studies (pp. 16–17). That mismatch is repairable by wording and provenance citation, but it matters because prospective status is part of the paper's argument for evidentiary weight.

### D4 — cross_disciplinary_relevance: `warn`

The paper makes real contact with active curriculum learning, PLR, posterior selection, bandit-style allocation, and systems cost rather than relying on analogy (pp. 7–8, 13–17). The remaining weakness is that the count-law teacher is an adaptive partial-feedback system, yet the deployment discussion does not fully expose the resulting feedback loop: a unit's estimated count law is learned only when that unit is selected, while decay, concentration, and the uniform floor are left as empirical choices (Algorithm 1 and instantiation, p. 5). The imperative recommendation to use a score exponent at least as large as deployed group size (p. 8) is also stronger than the evidence base, which is explicitly small, co-designed, platform-sensitive, and unable to locate long-horizon utility from activity alone (pp. 6, 9, 15). These are bounded transfer and systems-feasibility issues, not failures of the theorem or P0 intervention.

### D5 — writing_and_structure: `pass`

Despite high information density, the main text states one central claim early, leads with the same-mean counterexample, interprets each theorem in prose, separates evidence types, and keeps the conclusion within page 9. Figures 1, 2, 4, and 5 make the unit mismatch and evidence perimeter easy to recover (pp. 1–3, 7, 13–15). The appendices are dense but generally label exploratory, descriptive, inconclusive, and externally unavailable records at the point of use. Minor terminology and notation clarifications would help, but presentation does not impede evaluation.

### F0–F3 inputs and checks

- **F0:** `false`. D1 and D2 are `pass`, but mandatory D3 is `warn`; therefore the all-mandatory-pass condition for `accept` is not met.
- **F1:** `false`. No mandatory dimension is `block`, so `reject_or_major` is not triggered.
- **F2 local predicate:** `false`. Exactly one mandatory dimension (D3) is `warn` or worse; this reviewer therefore does not contribute to the panel count requiring two or more mandatory dimensions at `warn` or worse. The final five-reviewer count and strict threshold of 4 belong to synthesis.
- **F3:** `false`. D4 is `warn`, not `block`, so the D4-triggered `major_revision` rule does not fire.

## Review Body

### Reviewer identity and confidence

I am reviewing as an active-learning, bandits, adaptive-data-selection, and ML-systems researcher. I regard myself as an adjacent-field reviewer rather than the final authority on MaxRL conventions. Confidence: **4/5**.

### Summary assessment

The strongest idea is not a new acquisition curve but a decision-unit correction: the estimator consumes a success-count law, while a curriculum may pool observations at a coarser unit, and averaging before applying the estimator's coefficient map need not agree with applying the map before averaging. That distinction is immediately recognizable from active learning and bandits, where acquisition statistics are only meaningful relative to the observation unit and feedback process. Here it is made exact for practical MaxRL, then tested with a particularly clean same-substrate intervention.

I also value the negative evidence. The manuscript shows that activity can rank continuation utility without locating its optimum, is information-starved as a one-Bernoulli replacement for a per-timestep PLR signal, and can lose under coarse pooling or against uniform sampling. This is a credible boundary-mapped paper, not a universal scheduler paper.

My requested revisions do not require new experiments. First, align “registered/confirmed” language with the provenance actually available. Second, present the online count-law teacher more explicitly as a partial-feedback, nonstationary selection system. Third, narrow the recommendation that the score exponent should be at least the deployed group size: the theorem identifies the estimator-matched shape, but the empirical evidence does not establish a general lower bound on the best curriculum hardness.

### Strengths

1. **The statistic is matched to the consumed unit.** Figure 1 and Corollary 2 turn a common aggregation mistake into an exact, auditable distinction (pp. 1, 5). This is valuable beyond RLVR because adaptive selectors routinely confuse a bucket-level mean with the distribution of atomic outcomes.

2. **The intervention isolates the correction that the theory motivates.** P0 varies the plug-in versus count-law score on one substrate while holding the rest of the training system fixed, and the paper avoids calling its per-level secondary a mediation test (pp. 7–8, 15).

3. **Negative results establish operational boundaries.** The PLR result explains why terminal binary feedback cannot simply replace a dense critic signal (p. 7); Digits rejects an estimator-matching learning law (p. 16); and the allocation studies show realized coefficient mass can move opposite downstream performance (pp. 17–18).

4. **The systems accounting is unusually candid.** Paid probes, optimizer-update dose, overshoot, external raw size, unavailable checkpoints, and blocked robustness replay are reported rather than hidden (pp. 16–19).

### Substantive issues

#### Issue 1 — Prospective-evidence labels exceed the visible timing record

- **severity:** `warn` (D3, argumentative coherence).
- **evidence:** The contribution list labels Acrobot “Registered and confirmed” (PDF p. 2, §1), whereas §3.1 calls its primary “internally frozen” (p. 6), §5 says registration timing for Acrobot and maze records is incomplete (p. 9), and Appendix D says some timing objects remain external (p. 19). Appendix C explicitly notes that several internal locks lack an immutable public pre-execution commit establishing timing (pp. 16–17).
- **reasoning:** An internal analysis rule may be scientifically useful, but “registered” ordinarily communicates prospectively fixed, independently auditable timing. The current wording asks the reader to grant more evidentiary status in the contribution list than the paper's own limitations can verify. This does not negate the effect estimate or design; it changes the appropriate verb and tier.
- **actionable fix:** Cite the immutable, independently timestamped preregistration for each study called “registered,” if one exists and is available under review. Otherwise replace “registered and confirmed” with a literal description such as “internally frozen and supported under its prespecified rule,” and reserve “preregistered” for studies whose pre-execution timing can be audited. Apply the same terminology consistently in the abstract, Figure 2 caption, contribution list, evidence section, and limitations.
- **confidence:** High.

#### Issue 2 — The adaptive partial-feedback loop needs an explicit systems contract

- **severity:** `warn` (D4, cross-disciplinary relevance and deployment feasibility).
- **evidence:** Algorithm 1 updates a unit only after observing a selected group's count and then samples from the resulting activity-weighted distribution (p. 5). The instantiation leaves floor, concentration, decay, and posterior estimator as empirical choices (p. 5). The deployment takeaway says the required statistics “come free from group outcomes you already log” (p. 8), while the appendix shows that information acquisition can dominate cost and dose when probes are needed (pp. 16–17).
- **reasoning:** From a bandit perspective, the count-law estimate is endogenous: low estimated activity reduces selection, which reduces fresh evidence, which can preserve stale or uncertain estimates as the policy changes. A uniform floor guarantees some visitation but does not by itself state a tracking rate, effective sample size, recovery behavior, or safe choice of decay. For MaxRL the arithmetic is cheap; reliable online identification is not automatically free. This matters most for large unit catalogs, drifting policies, asynchronous workers, and variable group sizes.
- **actionable fix:** Add a short operational contract separating computational sufficiency from statistical adequacy. State the per-unit state required by each estimator, the grouping key, fixed- versus variable-`N` behavior, update timing, minimum visitation implied by the floor, and what telemetry diagnoses stale count laws. Existing logs could optionally report effective decayed counts or the distribution of visits per unit; no new training run is needed. Scope “comes free” to fixed-`N` computation from already grouped outcomes, not to estimation quality or probing.
- **confidence:** High.

#### Issue 3 — “Use an exponent at least N” is a study-specific heuristic, not yet a deployment rule

- **severity:** `warn` (D4, transfer limitation).
- **evidence:** The takeaway instructs practitioners to score with `u_M` for `M ≥ N` and calls deployed `N` a floor (p. 8, §3.4). The supporting sweep rises through `u64` but falls at `u128`, uses a small fixed-pool setting, and shares seeds across portability runs (pp. 6, 15). The same section says activity “ranks utility well and locates it poorly” (p. 6), while §5 limits the evidence to one small Acrobot family and a co-designed sweep (p. 9).
- **reasoning:** The exact estimator-matched curve `u_N` and an empirically harder curriculum `u_M` answer different questions. Active-learning experience makes sharper acquisition functions particularly sensitive to coverage, drift, and model misspecification. The reported studies show that `M=N` is not an optimum and that harder can help at these operating points; they do not establish `M≥N` as a generally safe lower bound. The present imperative risks turning a carefully bounded paper into a universal tuning claim.
- **actionable fix:** Replace the imperative with a scoped heuristic: `u_N` is the estimator-matched starting hypothesis, while `M` is a hardness hyperparameter that may warrant a prospectively specified sweep under the deployment budget. State that the current evidence found improvement beyond `N` on the tested fixed-pool platforms but establishes neither monotonicity nor a transferable lower bound.
- **confidence:** High.

#### Issue 4 — Zero activity must not be read as zero task value

- **severity:** `warn` (D4, stakeholder and coverage implications).
- **evidence:** Figure 4 states that no priority rule can help outside the activity band and identifies relabeling as the creation channel (pp. 13–14); the main text correctly says activity is a gate rather than a learning law (pp. 8–9). Algorithm 1 includes a uniform floor (p. 5), but the broader coverage consequence of repeatedly downweighting all-fail units is not discussed.
- **reasoning:** In a deployed curriculum, currently impossible tasks can correspond to rare capabilities, underrepresented domains, verifier edge cases, or prerequisites that become learnable only after transfer. Estimator activity says such a unit emits no present within-group contrast; it does not say the unit is unimportant. Without an explicit coverage safeguard, an activity selector can encode a self-confirming curriculum frontier and systematically defer hard subpopulations.
- **actionable fix:** Add one limitations/deployment paragraph stating that zero activity is not zero scientific, safety, or stakeholder value. Recommend retaining exogenous coverage through the floor or a separate objective, logging exposure by task family, and treating creation/relabeling or prerequisite sequencing as separate mechanisms rather than silently dropping inactive units.
- **confidence:** Medium-high; the concern is operational rather than a defect in the stated theorem.

### Assumption audit

- **Explicit assumptions:** The paper is commendably explicit about binary verifiable rewards, fixed group size, estimator convention, permutation equivariance, and the additional conditional-i.i.d. assumption needed for a pass-rate-only curve (pp. 3–5). It also explicitly excludes gradient direction, optimizer state, transfer, and learning guarantees (pp. 4, 9).
- **Implicit assumption:** The deployment guidance assumes that the adaptive selector obtains sufficiently fresh count-law estimates under its own induced sampling distribution. The uniform floor and decay make that plausible in finite pools, but the paper does not state when the resulting estimates are reliable enough to drive selection.
- **Paradigmatic assumption:** The curriculum is primarily treated as a myopic acquisition rule over a fixed catalog. From nonstationary-bandit and ML-systems perspectives, the learner changes the reward law, the selector changes what is observed, and the task catalog or verifier may itself evolve. The manuscript recognizes pieces of this through decay and the activity-versus-utility distinction, but does not yet assemble them into an online-systems boundary.

### Cross-disciplinary connections and practical impact

The paper's most useful connection to active learning is a correction to uncertainty sampling: uncertainty in a bucket mean is not the same as expected acquisition value over atomic examples. Its connection to bandits is also sharper than the current prose makes explicit. Algorithm 1 is a nonstationary, structured bandit with endogenous feedback and forced exploration; coefficient activity supplies a model-based acquisition statistic, not a regret objective. Framing it this way would clarify both its value and its limits without claiming a new bandit theorem.

For fixed-`N` MaxRL, the implementation case is attractive: the zero-count frequency and mean count are compact per-unit sufficient statistics. RLOO additionally needs a second moment, while GRPO needs a count histogram (Algorithm 1, p. 5). The practical challenge is therefore less arithmetic than unit identity, state freshness, group assembly, and consistent logging across distributed workers. A one-table systems contract covering those items would materially improve transferability.

The likely stakeholders are model trainers, curriculum designers, benchmark/pool curators, and operators responsible for capability coverage. The paper should make explicit that the curator's choice of unit is itself consequential: pooling can erase usable signal, while overly fine partitioning can make estimates sparse. A deployment audit should therefore report both selection exposure and realized dead-group rates by meaningful task family, not only aggregate reward.

### Cross-disciplinary reading recommendations

- **Besbes, Gur, and Zeevi (2014), “Stochastic Multi-Armed-Bandit Problem with Non-stationary Rewards.”** The variation-budget perspective provides language for when decay and forced exploration can track a changing pass/count law.
- **Russo, Van Roy, Kazerouni, Osband, and Wen (2018), “A Tutorial on Thompson Sampling.”** Useful for separating posterior sampling as an exploration mechanism from a posterior draw passed through a task-value functional.
- **Settles (2009), “Active Learning Literature Survey.”** The uncertainty-sampling discussion gives a mature vocabulary for boundary focus, representativeness, and why a scalar acquisition score requires coverage safeguards.
- **Sculley et al. (2015), “Hidden Technical Debt in Machine Learning Systems.”** Its feedback-loop and data-dependency framing is directly relevant to per-unit teacher state, stale statistics, logging, and distributed deployment.
- **Perdomo et al. (2020), “Performative Prediction.”** The performativity lens helps articulate that training and selection change the future outcome distribution being estimated, even though the present paper need not solve that general problem.

### Questions for the authors

1. Which immutable timestamp or external record establishes the pre-execution status of each result called “registered,” especially Acrobot and P0?
2. What telemetry should a practitioner use to detect that a decayed count-law estimate is stale or supported by too few recent groups under the selector's own sampling distribution?
3. Is `M≥N` intended only as a summary of the tested Acrobot/CPU operating points, or as a general deployment recommendation? If the latter, what evidence makes `N` a lower bound rather than merely the estimator-matched reference point?
4. How should Algorithm 1 pool state when group size varies, prompts are deduplicated imperfectly, or asynchronous workers update the same unit at different policy versions?

### Minor issues

- On p. 8, “the preregistered MAZE-SCORE correction” could be renamed “the P0 count-law correction” so readers do not confuse it with the preceding MAZE-SCORE `u32` versus `p(1-p)` negative.
- Distinguish the finite-denominator stabilizer `ε` in §2 from the sampling-floor symbol in Algorithm 1; using different symbols would reduce implementation ambiguity.
- Define `M` explicitly as a score exponent—not a deployed group size—at the first `u_M` recommendation.

## Editorial Decision

**Minor Revision.** No `block` rule fires, and this reviewer's local F2 predicate is false, but F0 is unavailable because D3 is `warn`. The requested changes are bounded: align prospective-status wording with auditable provenance, qualify the exponent recommendation, and add an online-systems/coverage contract using existing definitions and artifacts. The final F2 panel count remains the synthesizer's responsibility.
