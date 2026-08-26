# Phase 1 Perspective Reviewer Precommitment

## Reviewer Identity and Scope

I will review from the perspective of an active-learning, bandits, adaptive-data-selection, and ML-systems researcher. The only paper metadata available in this phase are: target venue ICLR 2027; title *Score the Count Law, Not the Mean Pass Rate: Estimator Activity for Curriculum Selection in Verifiable-Reward RL*; field machine learning / reinforcement learning / adaptive data selection; approximately 12,000 words, 40 references, and 19 PDF pages, including 9 main-text pages. This document precommits evaluation criteria only and makes no claim about the manuscript.

## Contract Paraphrase

### D1: methodology_rigor — mandatory

I will assess whether the work's methods can support the claims they are used to support, viewed especially through adaptive sampling and sequential-decision concerns. The relevant standard is whether units, estimands, comparison conditions, feedback loops, dependence, uncertainty, and reproducibility are handled clearly enough that apparent gains cannot simply be attributed to selection effects, mismatched budgets, leakage, or implementation choices. Because D1 is mandatory, a foundational failure will receive `block`, a material but repairable deficiency will receive `warn`, and an adequately supported treatment will receive `pass`.

### D2: domain_accuracy — mandatory

I will assess whether concepts and claims are accurate within reinforcement learning and in the adjacent traditions of active learning, bandits, adaptive data selection, and ML systems. The standard includes precise definitions, correctly delimited assumptions, valid translations between adjacent frameworks, and claims of scope or novelty that do not erase materially related formulations. Because D2 is mandatory, a central domain error will receive `block`, an important correctable imprecision will receive `warn`, and domain-accurate treatment will receive `pass`.

### D3: argumentative_coherence — mandatory

I will assess whether definitions, assumptions, evidence, limitations, and conclusions form a traceable argument whose scope stays stable from motivation through implications. The standard is not mere internal fluency: conceptual distinctions must do real work, empirical or theoretical evidence must address the claims attached to it, and alternative interpretations must be acknowledged where they would change the conclusion. Because D3 is mandatory, a broken central inference will receive `block`, a consequential but repairable gap will receive `warn`, and a coherent argument will receive `pass`.

### D4: cross_disciplinary_relevance — high priority

I will assess whether the work makes meaningful contact with active learning, bandits, adaptive data selection, and deployment-oriented ML systems rather than relying on superficial analogy. The standard includes identifying which concepts genuinely transfer, where assumptions differ, what practitioners would need to implement the ideas, and which boundary conditions or unintended incentives matter under adaptive use. D4 has high priority: a failure that defeats the claimed cross-disciplinary relevance will receive `block`, a valuable but underdeveloped connection will receive `warn`, and a specific, appropriately bounded integration will receive `pass`.

### D5: writing_and_structure — normal priority

I will assess whether the presentation lets an expert reader recover the paper's central claim, assumptions, evidence hierarchy, and limitations within the stated page constraints. Definitions, notation, figures, section ordering, and terminology should reduce rather than create ambiguity, and the main text should expose load-bearing material instead of depending on hidden reconstruction. D5 has normal priority: presentation that prevents reliable evaluation will receive `block`, substantial clarity or organization problems will receive `warn`, and clear, navigable writing will receive `pass`.

## Rating Scale

- `block`: A dimension-level defect that undermines a central claim, prevents reliable evaluation, or cannot be repaired without substantial new analysis, evidence, or reframing.
- `warn`: A substantive, consequential issue that should be corrected but does not by itself invalidate the central contribution.
- `pass`: The dimension meets the relevant standard; any remaining comments are minor and do not change the dimension assessment.

No dimension will receive a label outside `block`, `warn`, or `pass`.

## Scoring Plan

### D1: methodology_rigor

**what_to_look_for:** I will look for explicit units of analysis and selection; alignment between theoretical objects, training-time statistics, and evaluation estimands; treatment of adaptivity, dependence, repeated use of observations, and selection-induced feedback; budget- and information-matched comparisons; baselines and ablations that isolate the claimed mechanism; uncertainty computed over the appropriate independent units; treatment-delivery checks; and enough implementation detail to distinguish an algorithmic result from a systems artifact.

**what_triggers_block:** I will assign `block` if a central conclusion relies on an unidentified or mismatched estimand, an invalid independence assumption, leakage between selection and evaluation, a fundamentally unmatched comparison, or missing evidence that would require a new study rather than a bounded repair. I will also assign `block` if the stated method cannot be reconstructed well enough to know what intervention was actually evaluated.

**what_triggers_warn:** I will assign `warn` if the core design remains interpretable but important safeguards, sensitivity analyses, baseline controls, uncertainty details, resource accounting, or implementation specifications are incomplete; or if some evidence supports a narrower claim than the one emphasized and the mismatch can be repaired by analysis or reframing.

### D2: domain_accuracy

**what_to_look_for:** I will look for precise and stable definitions; correct assumptions and boundary conditions; valid use of reinforcement-learning and adaptive-selection terminology; accurate distinctions among realized selection scores, expectations, estimator behavior, optimization signals, and downstream utility; careful handling of granularity and aggregation; and appropriately scoped relationships to active learning, bandits, curriculum learning, and data-selection mechanisms.

**what_triggers_block:** I will assign `block` if the primary result depends on a false domain claim, conflates non-equivalent objects in a way that invalidates the main conclusion, asserts a general result outside the assumptions needed for it, or builds its contribution on a materially incorrect account of an established adjacent formulation.

**what_triggers_warn:** I will assign `warn` if the main contribution can survive but important terminology, scope, equivalence claims, estimator interpretations, or relationships to adjacent fields are imprecise or overstated and require correction to prevent expert readers from drawing the wrong conclusion.

### D3: argumentative_coherence

**what_to_look_for:** I will map each central conclusion to its premises and evidence; check whether assumptions remain visible when results are interpreted; distinguish mathematical identity, empirical association, intervention evidence, and practical recommendation; examine whether alternative explanations are considered; and assess whether limitations constrain the same claims that the introduction and conclusion promote.

**what_triggers_block:** I will assign `block` if a main conclusion does not follow from the presented kind of evidence, if the argument changes the target quantity or scope between setup and conclusion, if an essential counter-interpretation is incompatible with the claimed result and unaddressed, or if the paper's central thesis depends on mutually incompatible premises.

**what_triggers_warn:** I will assign `warn` if the core argument is recoverable but one or more important links are implicit, evidence tiers are blurred, a causal or practical gloss exceeds the demonstrated result, limitations are not propagated to prominent claims, or plausible alternative interpretations need explicit adjudication.

### D4: cross_disciplinary_relevance

**what_to_look_for:** I will look for substantive connections to adaptive experimental design, active learning acquisition, bandit exploration, online data selection, and ML-systems constraints; explicit comparison of assumptions and objectives across those areas; implications for feedback loops, coverage, robustness, computation, logging, and deployment; acknowledgment of stakeholders such as model trainers, benchmark designers, and system operators; and follow-up questions that adjacent fields can actually test.

**what_triggers_block:** I will assign `block` if a central relevance or applicability claim rests on a cross-disciplinary analogy that fails under the adjacent field's basic assumptions, if the proposed interpretation would be operationally unusable without unavailable information while feasibility is claimed, or if ignoring adaptive feedback or deployment constraints reverses the practical meaning of the contribution. Under contract rule F3, any D4 `block` implies `major_revision`.

**what_triggers_warn:** I will assign `warn` if the work is potentially relevant across fields but the mapping is only partial, key assumption differences are not stated, practical feasibility or systems cost is underdeveloped, stakeholder consequences are omitted, or concrete borrowing opportunities are left at the level of broad analogy.

### D5: writing_and_structure

**what_to_look_for:** I will look for an early statement of the one central claim; definitions before use; consistent terminology and notation; plain-language interpretations of technical results; visible separation of claims by evidence type; legible figures and captions; a main-text structure that preserves load-bearing assumptions and results within nine pages; and an abstract and conclusion whose strength matches the body.

**what_triggers_block:** I will assign `block` if ambiguity, missing definitions, organization, notation, or dependence on inaccessible material makes the central contribution or its support impossible to evaluate reliably, or if contradictory framing prevents identification of the actual claim.

**what_triggers_warn:** I will assign `warn` if the paper can be evaluated but substantial compression, terminology drift, notation density, buried assumptions, weak signposting, figure opacity, or repetition materially impedes comprehension and can be repaired through restructuring or clarification.

## Required Issue Fields

Every issue reported in Phase 2 will contain all five of the following fields:

1. **severity:** The issue's consequence, stated consistently with the relevant `block`, `warn`, or `pass` dimension assessment.
2. **evidence:** A specific manuscript location, artifact, quotation kept within fair-use limits, result, or clearly identified absence that supports the issue.
3. **reasoning:** The explicit chain from the evidence to the concern and why it matters under this reviewer perspective.
4. **actionable fix:** A concrete revision, analysis, qualification, comparison, or implementation disclosure that would resolve or bound the concern.
5. **confidence:** My confidence in the issue, including any uncertainty arising from outsider status or field-convention differences.

## Decision Rules and Phase 2 Order

I will apply the contract rules as follows:

- **F1:** If any mandatory dimension—D1, D2, or D3—receives `block`, the decision is `reject_or_major`.
- **F2:** For each reviewer, two or more mandatory dimensions at `warn` or worse (`warn` or `block`) add that reviewer to the panel count. If the count reaches the strict five-reviewer majority threshold of 4, the decision is `major_revision`.
- **F3:** If D4 receives `block`, the decision is `major_revision`.
- **F0:** `accept` is available only if D1, D2, and D3 all receive `pass` and no higher-priority rule fires.

The required Phase 2 report order will be: **Dimension Scores** (D1–D5, followed by explicit F0–F3 evaluations), **Review Body**, then **Editorial Decision**. I will not silently change these precommitted triggers; if the governing protocol permits a scoring-plan dissent, I will disclose it before the Dimension Scores and within its stated limit.

## Paper-Blind Acknowledgement

This Phase 1 call remained paper-blind. I did not inspect the manuscript, repository, prior reviewer reports, `fable.md`, `AGENTS.md`, or any scientific artifact, and I have not inferred manuscript contents from the supplied metadata.

[CONTRACT-ACKNOWLEDGED]
