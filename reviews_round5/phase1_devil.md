## Contract Paraphrase

### D1: Methodology Rigor

D1 is mandatory and will be challenged at the level of whether the inferential design can actually support the claimed estimator/activity-to-curriculum link. I will test whether the objects being estimated, the unit of analysis, assumptions, comparisons, interventions, and validation checks line up with the scope of the conclusions. The eventual assessment must be `block`, `warn`, or `pass`; a D1 `block` activates F1, while a D1 `warn` or worse contributes to the reviewer-local F2 count.

### D2: Domain Accuracy

D2 is mandatory and will test whether reinforcement-learning and estimator concepts are defined and used correctly, whether mathematical claims have the assumptions they require, and whether the claimed novelty and relationship to adjacent curriculum methods survive domain-specific counterexamples. The eventual assessment must be `block`, `warn`, or `pass`; a D2 `block` activates F1, while a D2 `warn` or worse contributes to the reviewer-local F2 count.

### D3: Argumentative Coherence

D3 is mandatory and will stress every link from formal activity characterization through curriculum selection to any empirical or practical conclusion. I will look especially for hidden premises, shifts in the meaning or granularity of “activity,” causal overreach, and conclusions that outrun their evidence. The eventual assessment must be `block`, `warn`, or `pass`; a D3 `block` activates F1, while a D3 `warn` or worse contributes to the reviewer-local F2 count.

### D4: Cross-Disciplinary Relevance

D4 is high priority and will test whether the proposed framing remains meaningful across the theoretical, statistical, and reinforcement-learning viewpoints it invokes, without quietly relying on incompatible assumptions or terminology. The eventual assessment must be `block`, `warn`, or `pass`; under F3, any D4 `block` yields `major_revision` even if the mandatory dimensions do not independently trigger a higher rule.

### D5: Writing and Structure

D5 has normal priority and will test whether a roughly 12,000-word argument compressed into nine main-text pages remains traceable, scoped, and structurally honest. I will challenge presentation choices that hide assumptions, blur theorem/evidence/speculation boundaries, or make the central inference difficult to audit. The eventual assessment must be `block`, `warn`, or `pass`, but D5 alone is not a stated F1–F3 trigger.

The failure checks are binding. F1 has precedence whenever any mandatory dimension (D1–D3) is `block`, producing `reject_or_major`. For F2, this reviewer’s local predicate fires when at least two mandatory dimensions are `warn` or worse; the panel-level `major_revision` threshold is four of five reviewers meeting that predicate. F3 makes any D4 `block` a `major_revision`. F0 permits acceptance only when D1–D3 all `pass` and no higher-priority rule fires. The devil’s-advocate role will not supply a balanced recommendation or ordinal score, but Phase 2 will supply the D1–D5 signals and explicit F0–F3 inputs needed for panel synthesis.

Every eventual issue will contain exactly the five required evidentiary fields: `severity`, `evidence`, `reasoning`, `actionable_fix`, and `confidence`. Evidence will identify manuscript locations or reported results; reasoning will state why that evidence satisfies the precommitted trigger; the fix will describe a concrete remedy rather than merely restating the objection.

## Scoring Plan

### D1: Methodology Rigor

**what_to_look_for:** I will trace the claimed estimand from group-level outcome generation through coefficient computation, curriculum scoring, sampling or training intervention, and reported endpoint. I will inspect whether observational characterization is separated from causal intervention; whether task, rollout, group, seed, and evaluation units are kept distinct; whether comparisons isolate the statistic claimed to matter; whether assumptions match the data-generating process; whether uncertainty and multiplicity match the true replicate; and whether negative, bounded, or failed-delivery results constrain the conclusion. I will also look for direct tests that distinguish coefficient activity from optimization utility, learning progress, gradient quality, or long-horizon value.

**what_triggers_block:** `block` will require a foundation-level mismatch that invalidates the central inference—for example, the primary evidence cannot identify an estimator/activity-to-curriculum effect because treatment arms differ in uncontrolled ways; the effective replicate or outcome construction makes the reported uncertainty unusable for the principal claim; the activity measure is evaluated on a data-generating unit different from the unit consumed by the estimator without a justified bridge; or the methodology establishes only algebraic coefficient behavior while the central conclusion asserts learning utility. A block also fires if a required intervention or validation is absent and no narrower version of the principal claim remains supported.

**what_triggers_warn:** `warn` will fire when the central design remains interpretable but important qualifications are incomplete: limited regimes or seeds constrain generality; robustness checks do not cover plausible dependence, group-size, estimator, or optimization variants; intervention delivery or cost matching is imperfect but measurable; secondary analyses risk selective emphasis; or activity–utility distinctions are stated yet not consistently enforced in analysis and interpretation. These defects must be repairable by additional analysis, sharper scoping, or clearer reporting without replacing the core design.

### D2: Domain Accuracy

**what_to_look_for:** I will verify domain definitions and mathematical semantics for estimator coefficients, success-count laws, conditional independence, aggregation granularity, curriculum scores, and the relationship between realized group statistics and expected activity. I will test edge cases such as all-fail/all-success groups, dependence among rollouts, heterogeneous latent instances, variable group size, and alternative estimators. I will also examine whether adjacent methods already instantiate equivalent score shapes or decision rules and whether any claimed distinction is invariant to normalization, optimizer settings, and the chosen unit of aggregation.

**what_triggers_block:** `block` will fire if a core theorem, identity, or domain claim is false under its stated assumptions; if the manuscript conflates an arbitrary count law with a conditionally independent Bernoulli model in a way that drives the main result; if it attributes to activity a domain meaning such as gradient norm, information, or learning benefit that does not follow; if a standard or directly relevant existing method is mathematically equivalent to the proposed central mechanism, collapsing the claimed novelty; or if a plausible in-scope counterexample contradicts the headline claim and is neither excluded nor resolved.

**what_triggers_warn:** `warn` will fire for technically consequential but repairable imprecision: assumptions appear only after a result is used; notation shifts between atomic and aggregate units; normalization-dependent comparisons are presented too broadly; edge cases or estimator variants limit scope but do not falsify the stated theorem; terminology could mislead readers about what is measured; or related methods substantially narrow novelty without eliminating the surviving contribution. The issue must require explicit correction or scope reduction, not cosmetic wording alone.

### D3: Argumentative Coherence

**what_to_look_for:** I will reconstruct the argument as a chain of premises and conclusions: why mean pass rate is alleged to be insufficient, what the count law adds, how estimator-specific activity is derived, why a curriculum should respond to it, and exactly what empirical evidence supports beyond the algebra. I will seek non sequiturs, circular validation, equivocation across “score,” “activity,” “signal,” and “utility,” selective treatment of positive and negative evidence, and causal language unsupported by the comparison. I will formulate the strongest rival narrative and check whether it explains the same observations with fewer assumptions.

**what_triggers_block:** `block` will fire when the paper’s principal conclusion does not follow even granting the formal and empirical results—for example, an exact coefficient identity is used as sufficient justification for curriculum superiority; a same-mean counterexample establishes insufficiency of a statistic but is treated as proof that the proposed replacement improves learning; causal or general claims rely on descriptive associations; or acknowledged counterevidence directly defeats the advertised thesis while remaining outside the conclusion’s scope restrictions. An unresolved contradiction between theorem assumptions, experimental setting, and headline wording also blocks.

**what_triggers_warn:** `warn` will fire when the core claim can survive but the reasoning repeatedly invites a stronger interpretation than the evidence warrants: contributions mix proved, confirmed, descriptive, and open claims; alternative explanations are mentioned without being tested or bounded; the conclusion broadens beyond the theorem or intervention; negative findings are structurally marginalized; or key inferential transitions are implicit. A warn should be fixable through reorganization, explicit premise statements, claim-tier labeling, and narrower conclusions.

### D4: Cross-Disciplinary Relevance

**what_to_look_for:** I will test whether the paper translates correctly among probability/statistics, estimator theory, curriculum learning, and practical RL. I will ask whether each community can identify the object, assumptions, operational decision, and limitation; whether the formal measure connects to an actionable curriculum choice rather than remaining a renamed statistic; whether computational and data costs are part of that connection; and whether the framework reveals a transferable principle beyond one implementation without claiming universality.

**what_triggers_block:** `block` will fire if the cross-disciplinary bridge is essentially absent or materially misleading: the statistical object has no demonstrated or defensible operational connection to curriculum selection; the RL interpretation depends on assumptions foreign to the evaluated setting; the work’s relevance evaporates outside a single bespoke estimator–environment configuration despite broad framing; or terminology imports authority from another discipline while changing the concept’s meaning. Because D4 is high priority, such a block activates F3.

**what_triggers_warn:** `warn` will fire when a genuine bridge exists but important constituencies cannot safely transfer the result: assumptions are not translated into observable diagnostics; scoring cost or implementation constraints are ignored; applicability across estimator families, dependence structures, or task granularities is unclear; or the practical decision enabled by the theory is underspecified. These limitations must be addressable by a mapping table, worked examples, decision guidance, or explicit scope boundaries.

### D5: Writing and Structure

**what_to_look_for:** I will inspect whether the title, abstract, introduction, theorem sequence, evidence sections, limitations, and conclusion state one consistent claim at matching scope. I will look for early definitions, plain-language theorem interpretations, visible assumptions, clear separation of evidence tiers, faithful figure/table captions, and navigation between the nine-page main text and supplement. I will also test whether compression buries decisive caveats or distributes the same thesis across too many labels and subclaims.

**what_triggers_block:** `block` will be reserved for presentation failures severe enough to prevent evaluation or to systematically misrepresent the evidence: the central claim cannot be uniquely identified; essential assumptions or methods are inaccessible from the paper and supplement; notation or unit changes make derivations unauditable; or the abstract/conclusion assert a materially stronger result than the body while caveats are hidden in a way that changes the scientific meaning.

**what_triggers_warn:** `warn` will fire when the paper remains reviewable but structure predictably causes overreading or confusion: definitions arrive after use; theorem, empirical finding, and conjecture are not visually or verbally separated; the main text omits a necessary interpretive bridge; important negative evidence is hard to locate; dense compression obscures the contribution hierarchy; or terminology and symbols drift. The remedy should be a concrete reordering, cut, cross-reference, scope sentence, or notation repair.

Phase 1 remained strictly paper-blind: I did not inspect or infer any manuscript content, repository material, prior review, or scientific artifact.

[CONTRACT-ACKNOWLEDGED]
