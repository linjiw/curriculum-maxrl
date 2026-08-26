## Contract Paraphrase

**D1 — methodology_rigor.** From a domain-accuracy perspective, I will determine whether the study design, units of analysis, estimator implementation, data treatment, uncertainty reporting, and reproducibility materials are adequate for claims about group-relative policy-gradient estimators and curriculum selection in verifiable-reward RL. Because this is a mandatory dimension, a defect that invalidates the central inference cannot be offset by strengths elsewhere.

**D2 — domain_accuracy.** I will assess whether estimator formulas, curriculum concepts, verifiable-reward terminology, and empirical or theoretical claims match the current RL/RLVR evidence base; whether foundational and recent work is represented and attributed correctly; and whether distinctions among realized group statistics, population quantities, and learning utility are maintained. This mandatory dimension requires the contribution and its stated boundaries to be factually sound.

**D3 — argumentative_coherence.** I will test whether the central thesis remains logically consistent from definitions and assumptions through propositions, evidence, and conclusions, with each claim supported at the level asserted. In domain terms, identities, diagnostics, associations, interventions, and learning consequences must not be conflated; a central logical break is blocking because this dimension is mandatory.

**D4 — cross_disciplinary_relevance.** I will examine whether the framing and definitions allow readers in adjacent areas of machine learning, language-model post-training, statistics, and curriculum learning to understand the result and its applicability limits, and whether any transfer claims are supported rather than assumed. As a high-priority dimension, a foundational accessibility or substantiation failure can require major revision even if the core specialist discussion is sound.

**D5 — writing_and_structure.** I will assess whether the manuscript presents its question, notation, theory, evidence, limitations, figures, tables, and appendices in a clear sequence and follows ICLR conventions sufficiently for the contribution to be evaluated. This includes whether technical terms and visual encodings are introduced consistently and whether the main argument remains legible at the venue's expected level of compression.

## Scoring Plan

### D1: methodology_rigor

- `dimension_id`: `D1`
- `what_to_look_for`: Explicit definitions of the sampled unit, rollout group, reward variable, success count, estimator coefficient, curriculum score, and independent replicate; a study design that matches the claimed estimand; clear treatment of conditional-independence or dependence assumptions; faithful and comparable implementations of group-relative baselines; appropriate controls and ablations; separation of preregistered/confirmatory from exploratory evidence; uncertainty computed over genuinely independent units; transparent budgets, seeds, stopping rules, exclusions, and missing runs; and sufficient algorithms, code, configurations, artifacts, and derivations to reproduce the central findings.
- `what_triggers_block`: A central result depends on a design-estimand mismatch, estimator implementation error, leakage, post hoc endpoint or seed selection, pseudoreplication of correlated rollouts as independent evidence, or an unrecoverable absence of information needed to verify the core theorem or experiment; alternatively, causal or general learning claims rest only on an observational diagnostic with no design capable of supporting them.
- `what_triggers_warn`: The central inference remains interpretable, but one or more fixable deficiencies remain, such as incomplete reporting of seeds or budgets, unclear uncertainty conventions, limited robustness checks, an underexplained ablation, insufficient detail about dependence or missing-run handling, or reproducibility materials that are present but not yet straightforward to execute.

### D2: domain_accuracy

- `dimension_id`: `D2`
- `what_to_look_for`: Correct formulas and terminology for group-relative policy gradients, baseline estimators, finite-group coefficient behavior, verifiable binary rewards, pass-rate and pass-at-k quantities, and curriculum scoring; precise distinctions between realized count laws, Bernoulli or conditionally i.i.d. models, mixtures, and arbitrary dependent group laws; accurate separation of coefficient activity from gradient magnitude, variance, learning progress, or utility; and fair synthesis of foundational and recent work on PLR, SFL, adaptive curriculum/task sampling, GRPO/RLOO/MaxRL-style estimators, and adaptive rollout allocation, including exact equivalences or precedence that constrain novelty.
- `what_triggers_block`: The main theorem, novelty claim, or empirical interpretation relies on a materially incorrect estimator formula, false equivalence, invalid extension beyond stated probabilistic assumptions, conflation of coefficient activity with learning utility, or demonstrably false representation of decisive prior work; the error must undermine the paper's central domain contribution rather than a peripheral statement.
- `what_triggers_warn`: The main contribution remains domain-valid, but the paper omits or weakly integrates a relevant literature strand, attributes an idea imprecisely, uses inconsistent specialist terminology, leaves an applicability boundary implicit, overstates an incremental distinction, or contains a localized factual or mathematical imprecision that can be corrected without rebuilding the central claim.

### D3: argumentative_coherence

- `dimension_id`: `D3`
- `what_to_look_for`: A stable chain from research question to definitions, assumptions, theoretical statements, predictions or diagnostics, empirical tests, and conclusions; explicit qualifiers where results move from arbitrary group laws to conditionally i.i.d. or mixture settings; claim language calibrated to proof, controlled intervention, descriptive evidence, or open hypothesis; meaningful treatment of counterexamples and negative results; and conclusions that do not exceed what the cited evidence identifies.
- `what_triggers_block`: The core thesis contains an internal contradiction, the principal conclusion does not follow from the theorem or experiment offered for it, a counterexample defeats an unqualified central claim, or the argument repeatedly converts an estimator-side identity or correlation into a causal learning guarantee without supporting evidence.
- `what_triggers_warn`: The overall thesis is supportable, but a bridge between theory and experiment is underexplained, a qualifier is missing in a limited section, a secondary claim is stronger than its evidence, an alternative explanation is not discussed, or the organization temporarily obscures which evidence supports which claim.

### D4: cross_disciplinary_relevance

- `dimension_id`: `D4`
- `what_to_look_for`: Plain-language definitions and interpretations alongside specialist notation; explanations of why success-count distributions, estimator coefficients, and curriculum granularity matter to adjacent RL, LLM post-training, bandit/allocation, and curriculum-learning audiences; a clear account of which conclusions transfer across estimators, reward types, group sizes, and task aggregation regimes; and restrained interdisciplinary implications tied to actual theory or evidence.
- `what_triggers_block`: A cross-disciplinary claim essential to the stated contribution is unsupported or technically false, or core concepts are framed so idiosyncratically and left so undefined that adjacent-field readers cannot determine the scope, assumptions, or relevance of the result.
- `what_triggers_warn`: The specialist result is assessable, but adjacent-field access is hindered by unexplained notation or jargon, missing intuition, an unclear mapping to neighboring literatures, or transfer language that needs qualification concerning estimator normalization, group dependence, binary versus graded rewards, or task domain.

### D5: writing_and_structure

- `dimension_id`: `D5`
- `what_to_look_for`: A concise statement of the problem and contribution; early presentation of the key motivating distinction; consistent notation and terminology; theorem statements paired with intuitive interpretations; a coherent separation of theory, experiments, related work, limitations, and appendices; figures and tables with self-contained captions, readable labels, uncertainty definitions, and traceable quantities; and adherence to ICLR length, anonymization, citation, formatting, and supplementary-material conventions.
- `what_triggers_block`: Presentation defects prevent substantive evaluation—for example, the central claim cannot be located or reconciled across sections, essential definitions or results are absent, figures or tables carrying the argument are uninterpretable, or severe venue noncompliance makes the submission unreviewable.
- `what_triggers_warn`: The manuscript is reviewable but has correctable clarity or structure problems, such as delayed definitions, notation drift, redundant sections, weak signposting, crowded or insufficiently labeled visuals, ambiguous uncertainty labels, an overlong main narrative relative to the venue format, or minor citation and formatting inconsistencies.

[CONTRACT-ACKNOWLEDGED]
