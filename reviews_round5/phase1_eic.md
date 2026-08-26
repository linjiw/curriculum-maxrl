## Contract Paraphrase

**D1 — methodology_rigor.** From an editorial-oversight perspective, the submission must make its study design, data provenance and handling, statistical reporting, and reproducibility provisions sufficiently transparent and rigorous for peer review in machine learning and reinforcement learning. Because this dimension is mandatory, shortcomings that prevent readers from determining whether the central evidence is valid must be treated as potentially decision-blocking.

**D2 — domain_accuracy.** The submission must use reinforcement-learning and verifiable-reward terminology correctly, represent relevant prior work fairly, and keep its claims consistent with the current evidence base. As a mandatory dimension, a domain error or literature misrepresentation that changes the meaning, novelty, or validity of the central contribution cannot be treated as merely editorial.

**D3 — argumentative_coherence.** The central thesis must remain internally consistent across the manuscript, and each material claim must be supported by evidence of the kind and strength required for that claim. Since this dimension is mandatory, contradictions, unsupported inferential leaps, or scope inflation that undermine the paper's core argument are grounds for a blocking assessment.

**D4 — cross_disciplinary_relevance.** The framing, definitions, and implications should allow ICLR readers in adjacent areas to understand what transfers beyond the immediate specialty, while any interdisciplinary reach must be supported rather than asserted. This is a high-priority dimension: a failure that makes the claimed broader relevance materially unintelligible or ungrounded warrants strong editorial concern.

**D5 — writing_and_structure.** The manuscript should present its contribution in a clear, logically organized form, with legible and informative figures and tables and adherence to ICLR conventions. This normal-priority dimension concerns whether readers can efficiently recover the paper's thesis, evidence, boundaries, and implications from the approximately 12,000-word submission.

## Scoring Plan

### D1: methodology_rigor

dimension_id: D1

what_to_look_for: A clearly specified research question and unit of analysis; assumptions matched to design; traceable data-generation and preprocessing procedures; appropriate baselines, controls, ablations, and uncertainty reporting; independence and replication claims aligned with the actual experimental structure; separation of confirmatory and exploratory evidence; enough implementation, artifact, and evaluation detail to audit or reproduce the central results; and explicit treatment of limitations relevant to validity.

what_triggers_block: The paper's central claims depend on a design that cannot identify or test them, materially invalid data handling, statistical evidence whose sampling or replication unit is fundamentally misrepresented, missing provenance that prevents validation of load-bearing results, or an irreproducible central analysis with no adequate audit trail. The pattern must threaten the validity of the main contribution rather than reflect a localized omission.

what_triggers_warn: The core design appears interpretable, but important details, robustness checks, uncertainty characterizations, artifact links, baseline justifications, or reproducibility instructions are incomplete or ambiguous and could materially affect confidence without already invalidating the central evidence.

### D2: domain_accuracy

dimension_id: D2

what_to_look_for: Correct use of reinforcement-learning, estimator, curriculum-selection, and verifiable-reward concepts; precise distinctions among theoretical identities, modeling assumptions, empirical findings, and conjectures; faithful representation of closely related work; novelty claims calibrated to precedence; and domain-specific conclusions whose scope matches the cited theory and evidence.

what_triggers_block: A central theorem, empirical interpretation, or novelty claim rests on a false domain premise; key terminology is used in a way that changes the claimed result; decisive prior work is materially misrepresented or omitted such that the contribution is not original as framed; or factual errors invalidate the paper's main positioning or conclusion.

what_triggers_warn: The central contribution remains potentially sound, but literature coverage is incomplete, terminology or attribution is locally imprecise, distinctions among neighboring concepts need correction, or some domain claims extend modestly beyond the evidence and require narrowing.

### D3: argumentative_coherence

dimension_id: D3

what_to_look_for: A stable and explicit core thesis; alignment from title and abstract through contributions, results, limitations, and conclusion; evidence that directly addresses each principal claim; clear separation of coefficient-level, optimization-level, and downstream-learning inferences where relevant; stated assumptions and boundary conditions; and conclusions proportional to the strongest supported evidence.

what_triggers_block: The central thesis contradicts itself across sections, the main evidence does not bear on the principal claim, a necessary inferential link is absent, negative or boundary evidence defeats the stated conclusion without acknowledgment, or pervasive overclaiming makes the core argument unsustainable without substantial reconception.

what_triggers_warn: The overall thesis is recoverable and plausibly supported, but one or more material claims need qualification, the logical chain contains repairable gaps, evidence tiers are not consistently distinguished, limitations are not integrated into the conclusion, or framing overstates the demonstrated scope.

### D4: cross_disciplinary_relevance

dimension_id: D4

what_to_look_for: Definitions and motivating examples that are intelligible to adjacent ICLR communities; explanation of why the problem and contribution matter beyond a narrow implementation setting; disciplined translation between theory, estimator behavior, curriculum design, and learning implications; substantiated transfer claims; and a candid account of what does not generalize.

what_triggers_block: The paper's claimed broader or interdisciplinary significance is essential to its contribution but is unsupported, technically incoherent across fields, or inaccessible because foundational concepts and mappings are never defined, leaving the asserted relevance impossible to evaluate.

what_triggers_warn: The specialist contribution may remain valid, but adjacent-field accessibility is limited by missing definitions, insufficient motivation, unclear practical implications, or transfer claims that need evidence or narrower wording.

### D5: writing_and_structure

dimension_id: D5

what_to_look_for: A concise statement of the problem and contribution; coherent section order and signposting; consistent terminology and notation; readable proofs and empirical narratives; figures and tables that are legible, self-contained, and tied to claims; efficient use of the main-text budget; clear appendix routing; complete citations; and compliance with ICLR formatting, length, anonymity, and submission conventions.

what_triggers_block: Presentation defects are so pervasive that the core thesis or evidence cannot be reliably reconstructed, load-bearing figures or tables are uninterpretable, the manuscript materially violates venue requirements in a way that prevents review, or structural disorder obscures which claims are actually being made and supported.

what_triggers_warn: The paper is reviewable, but organization, notation, prose, figure/table exposition, redundancy, appendix navigation, or venue compliance needs substantial correction to meet publication standards.

[CONTRACT-ACKNOWLEDGED]
