# Skills learned from the MaxRL paper (arXiv:2602.02710)

A distillation of the writing, figure, and math presentation techniques that make
"Maximum Likelihood Reinforcement Learning" (Tajwar, Zeng, et al.) read so well.
Each skill is stated as a reusable rule, with the concrete place in the paper it
was learned from. Use these as a checklist when revising our draft.

---

## 1. Writing skills

### W1. One-sentence thesis, stated at three altitudes
The whole paper hangs on one sentence: *"Reinforcement learning optimizes a
first-order approximation of the maximum likelihood objective."*
It appears (a) in the abstract in plain words, (b) in §3 as a centered, italic
display line after the math earns it, and (c) implicitly in the conclusion.
**Rule:** if you cannot write your paper's thesis in one sentence, the paper is
not ready; once you can, repeat it at abstract / theory / conclusion altitude.

### W2. Display-line beacons
Twice in the paper, a single italic sentence is centered on its own line:
- *"Reinforcement learning optimizes a first-order approximation of the maximum likelihood objective."*
- *"MaxRL provides a principled framework for trading additional compute for higher-fidelity approximations to the maximum likelihood objective."*

These act as narrative beacons: a skimming reader who reads only the centered
lines and the takeaway boxes still gets the whole argument.
**Rule:** promote the 2–3 sentences that carry the argument to display lines;
everything else stays in paragraphs.

### W3. Claim-first paragraphs
Nearly every paragraph opens with its conclusion, then supplies evidence:
"All three objectives improve upon the base model, but the magnitude differs
markedly…" → then the numbers. Never the reverse.
**Rule:** first sentence = the claim; remaining sentences = support. A reader
who reads only first sentences should reconstruct the paper.

### W4. Contributions: three items, one sentence each, mapped to sections
Their contribution list is 3 numbered items, each a single sentence, each ending
with "(cf. Section X)". No caveats, no numbers, no sub-clauses inside the list —
caveats live in the sections themselves.
**Rule:** ≤4 contributions, ≤1 sentence each, each pointing at exactly one
section. If a contribution needs an embedded parenthetical correction or a
p-value, it is written at the wrong altitude.

### W5. Numbers in the intro are ratios and adjectives, decimals live in results
The abstract/intro use "up to 20× test-time scaling efficiency" and
"Pareto-dominates" — never ".279±.019 vs .274". Raw decimals appear only in §6
where the setup that gives them meaning has been established.
**Rule:** before §Experiments, only multipliers (20×), directions (grows/loses),
and scoped superlatives ("in all settings we tested"). After, full precision.

### W6. One coined term per paper
The paper coins exactly one name — MaxRL — and otherwise reuses standard
vocabulary (pass@k, coverage, REINFORCE, truncation order). Every new concept is
expressed through existing words plus the one name.
**Rule:** budget coined terms. Each new term is a tax on the reader; more than
~3 and reviewers start losing the thread.

### W7. Boxed, titled takeaways — one per experiment, claim in the title
Every experimental subsection ends with a shaded box: **"Takeaway 2: MaxRL
scales better with additional compute in the infinite data regime."** followed by
one elaborating sentence. The box *title* is the claim, not "Takeaway: summary".
**Rule:** one box per experiment; the title alone must be quotable in a review.

### W8. Scoped claims that preempt reviewers
"Pareto-dominates existing methods **in all models and tasks we tested**";
"One caveat: REINFORCE's failure maybe due to us training from scratch — on a
pretrained model, it indeed produces gradients but still shows poor gradient
norm". Strong claims carry their scope inline; known weaknesses get one honest
sentence in place, with details deferred.
**Rule:** attach scope to every superlative; give each known objection one
sentence at the point a reviewer would raise it.

### W9. The experiment ladder: escalating realism, one question per rung
§6 runs: (6.1) didactic setting where the ideal (exact ML) is computable →
(6.2) infinite-data regime → (6.3) data-scarce regime → (6.4) billion-parameter
reasoning models → (6.5) mechanism analysis. Each subsection opens by stating
the question it answers ("we evaluate how closely MaxRL approximates exact
maximum likelihood…"). The §6 preamble gives the map before the ladder starts.
**Rule:** order experiments from most-controlled to most-realistic; open each
with its question; open the section with a one-paragraph roadmap.

### W10. Honesty placed at the right altitude
Caveats appear exactly twice each: one sentence in the main text where relevant,
full detail in an appendix or the limitations discussion. The main results prose
is never interrupted by retraction narratives or lab-notebook history.
**Rule:** main text gets the *conclusion* of your self-audit; the appendix gets
the audit.

### W11. Generous, boxed appendices
Appendices contain: extended related work (organized by theme with topic-sentence
paragraph heads), full proofs, per-experiment hyperparameter *tables* (colored
title-bar boxes, two-column parameter/value layout), verbatim prompt templates in
labeled boxes, exact advantage formulas for every baseline, and pass@k estimator
formulas. Reproducibility is demonstrated, not asserted.
**Rule:** every number in the paper should be recomputable from an appendix
table; every baseline's exact update rule should be written out once.

### W12. Section names are plain
"Preliminaries", "Gradient Estimators for MaxRL", "Experiments", "Related Works".
No slogans as section titles — slogans live in display lines and takeaway boxes
where they do rhetorical work; titles do navigational work.

---

## 2. Figure and table skills

### F1. One color per method, frozen for the whole paper
MaxRL = red, GRPO = green, RLOO = blue, base/reference = dashed gray/black —
in *every* figure from Fig 2 to Fig 9. The reader learns the legend once.
**Rule:** build a method→color table before making any figure; the base or
reference condition is always a dashed neutral line.

### F2. Metric as panel title, with direction arrow
Panels are titled "Pass@1 (↑)", "Pass@128 (↑)", "−log(Pass@k) (↓ lower is
better)". The reader never has to guess whether up is good.

### F3. Panel grids sweep one variable
Fig 3 and Fig 4 are 1×4 grids sweeping k ∈ {1, 32, 128, 256}: same axes, same
curves, only k changes. The visual repetition *is* the argument (the effect
persists at every k).
**Rule:** when a claim quantifies over a variable, show a small-multiples grid
over that variable rather than picking one value.

### F4. Annotate the headline number inside the plot
Fig 5 draws dotted guide lines and prints "16.4×", "19.2×" inside each panel at
the point where the speedup is measured. The abstract's "up to 20×" is literally
visible in the figure.
**Rule:** the number in your abstract should be findable, printed, inside a
figure. Limit in-plot callouts to 1–2 per panel.

### F5. Curve families use sequential shading of one hue
Fig 1 shows the truncated objectives T ∈ {1,2,4,…,128} as progressively darker
oranges between two anchor curves (REINFORCE, exact ML in red; GRPO in green).
The gradient of shades *shows* the interpolation claim — no legend reading
required.
**Rule:** a parameter family = one hue, monotone lightness; anchors = distinct
saturated colors. Log–log axes when the behavior spans decades.

### F6. The one-figure theory
Fig 1 (weight functions w(p) vs pass rate, log–log) contains the entire theory:
RL is flat, GRPO bends and *inverts* at high p, MaxRL climbs toward ML as T
grows. Everything the math says is checkable by eye in one plot.
**Rule:** find the single plot that renders your theory falsifiable at a glance,
and put it next to the theory, not in the experiments.

### F7. Tables as arguments, not just data dumps
- Table 1 is 2 rows: the estimator formulas and the row **"Unbiased for"** —
  the paper's core conceptual distinction compressed to one table cell contrast
  (∇pass@1 vs Σ(1/k)∇pass@k).
- Table 2 is one row: w(p) for RL / GRPO / MaxRL(T) / ML — the unifying view.
**Rule:** design a table so the reader's eye makes the comparison you want;
the punchline should be a visible cell-vs-cell contrast.

### F8. Algorithm box with the changed line highlighted
Algorithm 1 prints a standard REINFORCE loop with the *single modified line* in
blue, matching the prose claim "a single-line modification". The claim of
simplicity is made visually verifiable.
**Rule:** if you claim a minimal diff to standard practice, render the diff.

### F9. Captions: bold name → what it shows → how to read it → what to conclude
E.g. Fig 2's caption names the setting, states what varies, and ends with the
conclusion ("REINFORCE fails to make meaningful progress even with very high
per-input sampling budget"). Captions are self-contained; a figure + caption
alone survives being extracted from the paper.

### F10. Qualitative-shape evidence: scatter + trend line
Fig 6/8 (gradient L² norm vs pass rate, one panel per method, fitted trend)
shows *shape* similarity between MaxRL and cross-entropy and shape difference
from GRPO (inverted-U). Mechanism claims are supported by distributional
pictures, not just endpoint metrics.

---

## 3. Math presentation skills

### M1. The expansion trick as narrative engine
The technical core is one identity, developed in three steps the reader can
follow line-by-line:
```
J_ML(x) = log p = −Σ_{k≥1} (1−p)^k / k = −Σ_{k≥1} fail@k(x) / k
⇒  ∇J_ML(x) = Σ_{k≥1} (1/k) ∇pass@k(x)          [boxed]
⇒  truncate at T:  ∇J^(T) = Σ_{k=1}^T (1/k) ∇pass@k ;  T=1 is RL, T→∞ is ML
```
The *only* boxed equation in the paper is the gradient identity — the pivot on
which everything turns. Truncation order T is then sold as "compute index",
converting an approximation parameter into a scaling story.
**Rule:** box exactly one equation; make the approximation parameter mean
something operational (compute, budget, samples).

### M2. Interpretation sentence after every display equation
No equation is left to speak for itself: "Thus, maximum likelihood optimizes an
infinite harmonic mixture of pass@k gradients, with higher-order terms encoding
rare success…". Prose restates each formula's meaning in words within one
sentence of its appearance.

### M3. Theorem in main text, proof in appendix, proof still elegant
Theorems 1–2 are stated in the main text with a one-line pointer to Appendix B.
The proofs themselves are short and use one clean idea each:
- Thm 1 (conditional form of the ML gradient): apply E[X | A] = E[X·1_A]/P(A)
  with A = {success}, so ∇J_ML = E[∇log m_θ(z|x) | success]. The success-
  conditioned expectation *is* the ML gradient.
- Thm 2 (estimator–objective equivalence): condition on K ≥ 1, note successes
  are i.i.d. draws from the success-conditioned law, multiply by P(K≥1) =
  pass@N, and telescope (∇p/p)(1−(1−p)^N) = Σ_{k=1}^N (1−p)^{k−1} ∇p.

### M4. "Unbiased for a different objective," not "biased"
The masterstroke framing: instead of apologizing that dropping K=0 groups makes
the estimator biased, they prove it is *exactly unbiased for the truncated
objective* J^(N). A defect becomes a definition. Corollary: for REINFORCE vs
MaxRL, "increasing N reduces variance of a fixed objective" vs "increasing N
improves the objective itself" — the same knob means different things per
method, and that contrast is the paper.
**Rule:** when an estimator mismatches the ideal objective, characterize exactly
which objective it does optimize; the characterization is usually more
interesting than the bias bound.

### M5. Control variates from first principles when standard arguments break
§4.2 notes the usual baseline argument fails (normalization by random K
correlates with the samples), so they use the unconditional mean score
V_N = (1/N)Σ∇log m_θ(z_i|x), which has exactly zero mean, and subtract it.
**Rule:** when you deviate from a standard variance-reduction recipe, say
precisely which assumption broke.
**Caveat we verified:** with the drop-both-terms-at-K=0 convention (their
Algorithm 1), E[V_N·1{K≥1}] ≠ 0, and the resulting estimator is exactly unbiased
for T = N−1 rather than T = N (symbolically checked:
Σ_{k=1}^{N}(1−p)^{k−1} − (1−p)^{N−1} = Σ_{k=1}^{N−1}(1−p)^{k−1}). Our draft's
Remark is correct — present it as a friendly refinement lemma with a 4-line
proof.

### M6. The unifying weight-function view
Express every method's population gradient as ∇J = E_x[w(p(x)) ∇p(x)] and
derive w(p) per method (Appendix C): RL → 1; GRPO → 1/√(p(1−p)); MaxRL(T) →
(1−(1−p)^T)/p; ML → 1/p. One functional form makes all methods commensurable,
turns "which method emphasizes hard prompts" into a plot, and yields free
corollaries (GRPO's weight *inverts* as p→1 — flagged as a possible cause of
sharpening, footnoted as conjecture).
**Rule:** find the common functional form; put the per-method derivations in an
appendix table; harvest the qualitative corollaries (limits, inversions, zeros)
in the main text.

### M7. Every quantity used gets a definition block
§2 Preliminaries defines pass rate, pass@k, the latent-generation model, and the
equivalence-relation caveat for answer matching before anything is claimed. Even
"all logarithms use base e" is stated. Appendices D.3–D.4 and G define exactly
how validation accuracy and pass@k are *computed* (including the unbiased
estimator with n ≥ k samples). No metric in any figure is undefined.
**Rule:** if a symbol or metric appears in a figure, its computation must be
written somewhere citable. Maintain a metrics paragraph.

### M8. Numerical sanity-checking as culture
Where exact computation is possible (ImageNet: pass@k computable analytically
from softmax probabilities), they compute exactly instead of sampling, and say
so. Estimates vs exact quantities are never silently conflated.
**Rule (extends to our practice):** MC-verify every closed form, but report the
verification in the appendix/repo — the proposition text itself should carry a
proof, not the label "MC-verified".
