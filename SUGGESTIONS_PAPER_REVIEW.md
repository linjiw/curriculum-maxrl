# Review of "The Estimator Decides" (draft v0.9) — suggestions to complete the story and lift the writing

Reviewed against: *Maximum Likelihood Reinforcement Learning* (arXiv:2602.02710), whose
craft is distilled in `SKILLS_FROM_MAXRL.md`. Rules cited as W#/F#/M# refer to that file.

---

## 0. Verdict in three sentences

The science underneath this draft is strong — the mass identity
u_N(p) = 2(pass@N − pass@1) is clean and correct (I re-verified it by Monte Carlo and
the T=N−1 remark symbolically), recycling-induced sharpening is a genuinely new
phenomenon, and the gate is a *derived* fix, which is rare. The writing currently buries
that science under lab-notebook honesty, ~12 coined terms, and abstract-level decimals;
a reviewer skimming for 10 minutes would come away with "complicated, hedged, two papers
stapled together." The fix is not more content — it is altitude control: one unifying
sentence stated early, decimals demoted to §7 and tables, caveats compressed to one
sentence each in place with the audit trail moved to an appendix.

---

## 1. The story is one sentence away from complete

You already have the unifier — it's Figure 3 — but it arrives at §6 in three sentences,
after the reader has struggled through the abstract. Promote it to the spine of the paper
(W1: thesis at three altitudes):

> **The estimator's advantage mass u_N(p) partitions task difficulty into a dead zone,
> a learnable band, and a mastered tail. Curricula can only reallocate compute within
> the band; recycling is the only channel into the dead zone; and relabels that land in
> the mastered tail buy sharpening, not signal. The estimator decides all three.**

With this spine:
- §3 (theory) *derives the partition* — the two zeros of u_N and its peak p* ≈ ln N/N.
- §4 (FrontierMax) = the band channel. §5 (recycling) = the dead-zone channel.
- Recycling-induced sharpening stops being a surprise bolted onto §7.6 — it is the
  *p→1 mirror image* of the p→0 story, predicted by the same functional. Say that in
  the intro, not just in §7.6.
- The gate is then "the functional applied a third time" — the paper's most elegant move.
  Currently that elegance is stated ("the same algebra supplies the fix") but not felt,
  because the partition was never installed as a mental picture.

Concretely: **merge Figure 3 into Figure 1** as a third panel or an annotated region
strip under panel A (dead zone / band / mastered, with "recycling acts here" and
"gate blocks here" arrows). That gives you the MaxRL-style one-figure theory (F6) on
page 3, and frees a figure slot. Then rename §6 from "Three channels, one safety rule"
(currently a 4-sentence orphan section) into a closing paragraph of §3 — a section that
short signals structural trouble to reviewers.

**Decide the headline.** Right now the title/abstract sell a *safety audit message*
("audit your estimator before you ship"), but the durable contribution is a *unified
theory of data-level interventions* with the audit as corollary. Recommend leading with
the theory (that is MaxRL's recipe: identity → family → predictions → confirmations)
and letting "audit your estimator" be the last line of the intro and the conclusion,
where slogans belong (W2).

---

## 2. Top 10 issues, prioritized

### P1. Rewrite the abstract (highest leverage)
Current: ~450 words, one paragraph, containing "permutation p=0.0079, nine runs, zero
exceptions", ".279±.019 vs. no-recycling .274", "~60%", "+.016 of +.026", a replication
narrative, and a parenthetical correction to another paper. That is §7 material at
abstract altitude (W5). MaxRL's abstract: ~200 words, zero decimals, one coined term.

Draft replacement (~190 words — edit freely, keep the shape):

> Post-training with verifiable rewards increasingly relies on two data-level
> interventions: difficulty curricula, which reallocate rollout budget toward learnable
> prompts, and failure recycling (hindsight relabeling), which converts failed rollouts
> into verified successes of the tasks they actually achieved. Both are treated as
> objective-agnostic add-ons. We show they are not: the advantage estimator underneath
> decides whether each helps or harms, in closed form. For success-conditioned (MaxRL-
> style) advantages, the expected advantage mass a prompt emits from N rollouts is
> exactly 2(pass@N − pass@1) — a compute-indexed learnability functional whose band
> location, width, and scaling follow from the rollout budget alone, and whose two zeros
> partition difficulty into a dead zone no sampler can reach and a mastered tail where
> updates buy sharpening instead of signal. This predicts an estimator-conditioned
> divergence we confirm across four task suites: under identical curricula, MaxRL-style
> runs grow pass@k coverage while GRPO runs lose it. It also predicts a failure mode of
> recycling we observe in production — hindsight arms improve mean accuracy while losing
> coverage — and supplies the mitigation: gating relabel admission by the destination's
> pass rate converts the loss into a tunable mean-versus-coverage dial. One diagnosis
> covers both: audit the estimator before shipping a curriculum or a recycler, and
> measure coverage, the currency in which both failure modes are visible.

Everything cut (p-values, seed counts, the GSM8K replication status, the ±'s) reappears
in §7 where it has context. "Confirm across four task suites" is the honest
abstract-level summary; the LLM-scale caveat gets its one sentence in §7.5 (W8, W10).

### P2. Cut coined vocabulary from ~12 terms to 4
Count in the current draft: advantage mass, ZPD functional, FrontierMax,
recycling-induced sharpening, three channels, dead zones, frontier amplification,
the escalating ladder, honest nulls, posterior-starved, dose-vs-direction, the gate,
steering, tier. Each is a tax (W6). Keep four: **advantage mass**, **FrontierMax**,
**recycling-induced sharpening**, **the gate**. Replace the rest with plain phrases
("the low-p region no sampler reaches", "gain from extra gradient dose vs. relabel
direction", "curriculum strength"). "Zone of proximal development" — use once, as a
parenthetical gloss, not as a load-bearing noun phrase.

### P3. Contributions: 5 multi-clause items → 4 single sentences (W4)
Current item 1 contains a formula, a scope remark, *and* a correction to another paper.
Proposed:

> 1. **A closed-form curriculum.** The expected advantage mass of a prompt under
>    success-conditioned RL is exactly 2(pass@N − pass@1), a compute-indexed
>    learnability functional whose band, width, and N-scaling are derived, not tuned
>    (§3); published learnability curricula are its N=2 slice.
> 2. **A derived teacher and a characterized recycler.** FrontierMax schedules prompts
>    by this functional (§4), and hindsight relabeling with a verified-success contract
>    yields the ML gradient of the achieved task under the original sampling law (§5).
> 3. **Estimator-conditioned predictions, confirmed.** Under identical curricula,
>    success-conditioned runs grow pass@k coverage while GRPO runs lose it, across four
>    task suites from exact-gradient chains to LLM post-training (§7).
> 4. **A new failure mode and its derived fix.** Exact-verifier recycling improves mean
>    accuracy while losing coverage — recycling-induced sharpening — and gating relabels
>    by destination pass rate converts the loss into a monotone mean-vs-coverage dial (§7.6–7.7).

The T=N−1 point moves to a Lemma in §3 (see P8). The "honest evidence discipline" item
is not a contribution — it is your methodology; put one sentence in §7's preamble and
the audit details in an appendix (W10).

### P4. Move the lab notebook to an appendix
The draft narrates its own history: retractions, an epoch-frozen sampler bug, a
replication seed with weaker steering, "a status we report as measured", "our own first
version silently used per-bin parameters and reproduced the known failure." This
honesty is a genuine asset — MaxRL does it too — but at one sentence per caveat in main
text, with the full postmortems in an appendix ("Appendix H: Negative results and
audit trail", built from EVIDENCE.md / E_LLM1_POSTMORTEM.md). Currently §7.5 spends
~350 words on the replication story and §9 is a wall of nine caveats in one paragraph.
Restructure §9 into 4 themed short paragraphs: *Scale* (GSM8K interaction is 1-of-2
seeds, treatment-intensity-dependent; decisive steering-controlled cell queued),
*Scope of the surrogate*, *Environment-dependence of relabel exactness*,
*What did not transfer* (γ-concentration, duration hypothesis).

### P5. §7's rungs each need Setup → Result → Takeaway discipline
The ladder concept is good (W9) and the takeaway boxes are good (W7) — keep both. Fix:
- Open §7 with a roadmap paragraph + **one results overview table** (rung, setting,
  scale, seeds, headline metric, outcome) so a reviewer can see the whole ladder at
  a glance before descending.
- Rename rungs as the *questions* they answer: "7.1 Does allocation saturate?",
  "7.2 Can any curriculum act when nothing is learnable?", "7.3 Does the estimator
  main effect survive real gradients?", "7.4 What does a curriculum add beyond task
  spread?", "7.5 Does it transfer to LLM scale?", "7.6–7.7 What does recycling cost,
  and does the gate recover it?"
- Fix the numbering: 7.4a/7.4b → 7.4 and 7.5, renumber the rest. Sub-lettered
  paragraphs read as late insertions.
- Provenance notes ("the maze teacher predates the exact derivation and uses the legacy
  frontier form…", total-variation 0.013 at N=8) → appendix. One sentence in main text:
  "The maze teacher uses the legacy frontier form; the two forms' sampling
  distributions are within-noise equivalents (App. X)."
- Statistics: state the protocol once in §7's preamble (seeds, permutation tests,
  what ± denotes — SD or SEM, currently never defined), then stop re-litigating
  per-number. "(though that arm is single-seed)" style parentheticals — one per
  subsection maximum, rest to the table footnotes.

### P6. Freeze a paper-wide visual identity (F1)
Current inconsistencies: Fig 1B has MaxRL=blue, GRPO=pink-dotted, RLOO=green-dashed;
Fig 4a–c use green=uniform / orange=champion; Fig 4d uses pink=GRPO / blue=MaxRL;
Fig 6 has GRPO=green, GRPO+teacher=pink; Fig 5 uses orange=FrontierMax, green=uniform.
The same method changes color four times. Proposal:
- **Hue = estimator**: MaxRL family = blue, GRPO = red/pink, RLOO = green,
  baseline/uniform = neutral gray (dashed where it's a reference line).
- **Line style or marker = intervention**: uniform solid, +teacher dashed,
  +recycling dotted, +gate dash-dot. Then Fig 6's four cells and Fig 4's bars need
  no per-figure legend relearning.
- Recycling/gate figures (Fig 7) may keep the orange accent for the gate as the one
  intervention-specific color — but then the gate is orange *everywhere*.

### P7. Per-figure fixes
- **Fig 1**: your best figure — keep. Add "(↑ more learning signal)" to the y-label
  (F2). Merge Fig 3 into it (see §1). Panel B's three-way comparison is the
  paper's Table-2-equivalent; consider adding the tiny one-row table of u(p) formulas
  under it (F7): MaxRL 2((1−(1−p)^N)−p) · RLOO 2p(1−p) · GRPO (1/N)E√(K(N−K)).
- **Fig 4**: four bar panels with four different metrics and four different color
  schemes. Unify colors per P6, add metric arrows "AUC (↑)", and consider dot+CI
  instead of bars (3–5 seeds each; bars hide the seed count, dots show it).
  Panel (d) duplicates Fig 6 — pick one home for GSM8K in the main text.
- **Fig 5**: good — the 60×/13× in-plot callouts are exactly MaxRL's F4 move. Align
  the two x-axes (one ends at 600, other at 640 with odd ticks 210/430), and put
  "(↑)" on the y-label.
- **Fig 6**: three checkpoints per curve makes an LLM-scale claim look thin. If
  step-level val logs exist, plot every eval point with the divergence window shaded;
  if not, replace with a 2×2 table (start/mid/final per cell) and keep the prose.
- **Fig 7**: strong mechanism triptych (F10) — keep. In (a), label axes with direction
  ("coverage (↑)" / "mean@16 (↑)") so the trade-off's geometry is instant; annotate
  the arrow from B2→B3 with "gate".
- **Missing figure the story wants**: a pass@k-vs-k small-multiples or single panel
  (k = 1…16) for Countdown B1/B2/B3 (F3). Your claim — sharpening = mean up, coverage
  down, gate restores — *quantifies over k*; one panel with three curves crossing
  makes it unforgettable and mirrors how Yue et al. made the RLVR-sharpening point.

### P8. Math presentation (M1–M8)
- **Box exactly one equation**: the mass identity in Prop 1. It is your ∇J_ML identity
  (M1) — the thing a reader should remember.
- Remove "MC-verified" from proposition *titles* (Prop 1's title currently reads
  "Advantage mass; MC-verified"). Propositions carry proofs; verification scripts are
  a Reproducibility bullet. The 3-line proof of Prop 1 belongs in an appendix:
  Σ|w_i| = K(1/K − 1/N) + (N−K)/N = 2(1 − K/N) on {K ≥ 1}, and
  E[2(1 − K/N)1{K≥1}] = 2(P(K≥1) − p) = 2(pass@N − pass@1). Short, so show it.
- **The T=N−1 remark**: promote from a clause inside contribution 1 to
  "Lemma 1 (Truncation order of the practical estimator)" with its 4-line telescoping
  proof, phrased as a refinement, not a correction: "the drop-K=0 variant of Eq. (10)
  of Tajwar et al. is exactly unbiased for J^(N−1); the distinction is immaterial for
  their conclusions but matters for our mass accounting at small N." (Verified
  symbolically: Σ_{k=1}^{N}(1−p)^{k−1} − (1−p)^{N−1} = Σ_{k=1}^{N−1}(1−p)^{k−1}.)
  Consider emailing the authors — good citation karma.
- **Reconcile mass with the weight-function view** (M6). MaxRL's Table 2 gives
  w(p) = signed weight on ∇p; your u_N(p) = expected |advantage| mass. They are
  different objects and a reviewer who knows both papers will ask. Add a remark:
  u_N(p) = 2p(1−p)·w-something? — state the exact relation (for MaxRL,
  u_N(p) = 2(pass@N − p) while w_N(p) = (1−(1−p)^N)/p; note u_N(p) = 2p·(w_N(p)·p
  correction…) — derive it once cleanly) and *why* mass is the right sampling
  utility (your Remark 1 already argues this — link the two explicitly).
- **Interpretation paragraphs after propositions**: you already do this well (M2) —
  keep, but trim each to ≤3 sentences; Prop 3's interpretation currently smuggles in
  an experimental result (§7.1 oracle tie), which belongs in §7.
- **Define every metric before first figure use** (M7): AUC (of which curve, over
  what x?), mean@16, coverage ≡ pass@k at which k per suite, "tier", "frontier",
  "steering", "dose". Add a Metrics paragraph to §2. §2 "Background: MaxRL in three
  lines" should grow into a real Preliminaries: notation, pass@k, the three advantage
  formulas (RLOO/GRPO/MaxRL) side by side — mirror MaxRL's App. F but compressed —
  so §3's comparisons have referents.

### P9. Add the two standard artifacts reviewers scan for
- **Algorithm box** for FrontierMax (F8): a standard GRPO-style training loop with the
  three modified lines highlighted (Thompson-sample prompt by u(p̂); K=0 → relabel with
  gate; posterior update on requested task only). §4 is currently prose; the "honest
  knob inventory" becomes a 4-row table (knob, default, derived-or-tuned, sensitivity).
- **Hyperparameter tables per suite** in appendices (MaxRL App. E style), plus task
  format boxes (a Countdown example with a relabel, a maze instance). Your repro
  statement points at the repo — good, but the paper itself should carry the tables
  (W11). Also state compute (GPU type × hours per rung).

### P10. Sentence-level pass
- **Em-dash chains**: the draft habitually nests two or three "—" clauses per sentence
  (the abstract's fourth sentence has three, plus two parentheticals). Budget: one
  em-dash pair per sentence; split the rest. Read §7.3's first sentence aloud —
  it cannot be parsed in one breath.
- **The word "honest"** appears in section titles, paragraph heads, and prose
  ("honest nulls", "honest evidence discipline", "honest 0.5× reversal", "honest knob
  inventory"). Show it, don't label it — after the first use it reads as protesting
  too much. Same for "papered over".
- **Present results without hedging stacks**: "(0.308 → 0.271 uniform; 0.332 → 0.269
  with the teacher, the largest observed decay, though that arm is single-seed)" —
  move the qualifier chain into the results table.
- Title: fine, but consider dropping the second em-dash clause:
  "The Estimator Decides: What Curricula and Failure Recycling Can and Cannot Do in
  RL with Verifiable Rewards."

---

## 3. What to keep — these already match the good paper
- Figure 1 (the one-figure theory), the takeaway boxes, the Q1/Q2/Q3 intro frame,
  the escalating-ladder concept, the compute-cost hook that opens the intro
  (87% of step time / 65–75% zero-signal groups — best paragraph in the draft),
  the sharp closest-neighbor paragraph in Related Work (LfH), the Interpretation
  paragraphs, and the pre-registration discipline itself. Don't sand these off.

## 4. Ordered work plan
1. Rewrite abstract (P1) and contributions (P3). ~2 hours, changes everything a
   reviewer sees first.
2. Merge Fig 3 → Fig 1; install the partition sentence in intro + §3; dissolve §6. (P0/§1)
3. Vocabulary purge (P2) and em-dash/hedging pass (P10).
4. §7 restructure: preamble + overview table + question-named rungs + stats protocol
   paragraph (P5); move provenance/postmortems to Appendix H (P4); rewrite §9 as four
   themed paragraphs.
5. Figure palette freeze + per-figure fixes + the pass@k-vs-k Countdown panel (P6, P7).
6. Preliminaries upgrade, metric definitions, Lemma for T=N−1, boxed mass identity,
   proofs appendix, algorithm box, hyperparameter tables (P8, P9).
7. If at all possible before submission: run the queued steering-controlled multi-seed
   GSM8K cell. The draft itself names it "the decisive next experiment" — a reviewer
   will ask why it wasn't run. If it lands, §7.5 stops needing 350 words of hedging;
   if it nulls, the paper's maze-anchored main claim still stands and you report it
   in one sentence.

## 5. Verification notes from this review
- Prop 1 closed form re-verified by Monte Carlo at (p,N) ∈ {(.05,8),(.2,16),(.5,4),(.9,32)}:
  matches 2(pass@N − pass@1) to MC error.
- T=N−1 unbiasedness identity re-verified symbolically (SymPy).
- Prop 2 limit checked analytically: MaxRL mass ~ 2p(N−1), RLOO mass ~ 2p as p→0 ⇒ ratio → N−1. ✓
