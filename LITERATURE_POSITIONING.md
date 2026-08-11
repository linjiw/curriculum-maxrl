# Literature Positioning & Draft-Polish Guide

**Date:** 2026-08-11 · Companion to `claude-fable-plan.md`
**Inputs:** three parallel web surveys (RLVR curriculum/estimator analyses; robotics curriculum + BARN; HER/pass@k lineage), checked against the compact draft's related work (`origin/codex/curriculum-maxrl-research:paper/body_iclr.tex` §Related work) and `curriculum_maxrl/RESEARCH.md`.
**Verification status:** entries marked ⚠ were verified from arXiv abstract pages only, by agents. Before any citation enters the bib: open the abstract page, confirm authors/venue, and full-text-read the four marked MUST-READ.

---

## 1. Novelty verdict (the thing to protect)

**The identity survives.** No found paper derives the *unnormalized* expected
absolute coefficient mass `A_N(p) = 2(pass@N − pass@1)` with the
budget-dependent peak `p* ≈ ln(N)/N` and `p(1−p)` as the N=2 slice.

**But the neighborhood is now crowded (2025–2026), on a different branch:**
the field's closed forms live on the **std-normalized** branch — expected
GRPO update magnitude `√(k(G−k))/G` → `√(p(1−p))` — which is N-flat (or only
finite-G-corrected). Our identity is genuinely N-dependent and gives pass@N
an interpretation *inside* the estimator rather than as an external objective
(PKPO, Pass@k Training) or a diagnostic (Yue et al.).

**The one-line wedge (use everywhere, both venues):**

> *The same estimator that trains the policy prices the tasks — we take the
> deployed estimator's own algebra as the selection utility, rather than
> importing a heuristic (p(1−p), difficulty bands, learning progress) or
> rewriting the objective (pass@k rewards).*

**Concessions to make voluntarily (cheaper than a reviewer extracting them):**
- The **Beta-posterior + Thompson teacher mechanism is prior art** (MoPPS,
  arXiv:2507.04632 ⚠). Claim novelty only in the *utility being sampled
  toward*, never in the bandit machinery.
- Closed-form |advantage|-vs-p analyses of group estimators now exist
  (arXiv:2607.00152, 2602.20532 App. A.2 — both `2√(p(1−p))`-family ⚠).
  Cite as concurrent; differentiate on unnormalized + N-dependent + pass@N
  interpretation + curriculum use.
- "Dropping all-fail groups changes the objective": arXiv:2607.00152 ⚠
  already quantifies DAPO's discarded mass as `p^G+(1−p)^G`. Our surviving
  distinct claim is the **truncation-order T=N−1 result for the MaxRL
  Maclaurin convention** — keep that framing exact.

---

## 2. Threat matrix — top items, with differentiation one-liners

| # | paper | overlap | our differentiation (one sentence for the paper) |
|---|---|---|---|
| 1 | **Group-Std Identity** (Bay, arXiv:2607.00152 ⚠ MUST-READ) | exact E[update magnitude] vs (p,G) for GRPO/Dr.GRPO/DAPO; DAPO discarded mass | "Concurrent work characterizes the *std-normalized* update magnitude, whose expectation is N-flat `√(p(1−p))`; the unnormalized coefficient mass of the practical MaxRL convention is instead exactly `2(pass@N−pass@1)`, whose peak moves with the rollout budget — the property our teacher exploits." |
| 2 | **Actor-Curator** (arXiv:2602.20532 ⚠ MUST-READ) | bandit curriculum + closed-form E\|A\| = 2√(p(1−p)) (their beaten baseline) | "Treats N-independent absolute advantage as a baseline; our N-dependent identity explains when that baseline's peak is misplaced." |
| 3 | **MoPPS** (arXiv:2507.04632 ⚠ MUST-READ) | streaming Beta posterior + Thompson prompt selection targeting p≈0.5 | "We adopt the same posterior machinery; the contribution is the utility — estimator-exact and budget-aware rather than a fixed intermediate-difficulty target." |
| 4 | **Learning-Zone Energy** (arXiv:2605.17003 ⚠) | closed-form score "aligned with expected GRPO update magnitude" | same wedge as #1: p(1−p)-variance-based, N-flat. |
| 5 | **F-GRPO** (arXiv:2602.06717 ⚠) | closed-form N-dependent group-composition analysis tied to pass@k collapse | "Analyzes rare-mode coverage loss and patches the weighting; we characterize where mass lands and select tasks, leaving the estimator intact." |
| 6 | **SFL** (arXiv:2408.15099) + **LILO** (arXiv:2502.12272) + **ProCuRL** (arXiv:2304.12877) | p(1−p)-shaped scores (heuristic / idealized-learner-derived) | already cited; keep the "N=2 slice; derivation from the *deployed* estimator; peak moves with N" line. ProCuRL is the strongest prior "derived not heuristic" claim — name its derivation source (idealized-learner value improvement) explicitly. |
| 7 | **LfH** (arXiv:2607.09042 ⚠ MUST-READ) + **AgentHER** (arXiv:2603.21357 ⚠) | group-based failed-rollout relabeling for VLAs / LLM agents | already cited (lfh); press: judge-scored vs verifier-exact; SFT/DPO-offline vs on-policy group-estimator loop; no estimator-derived placement rule; no coverage accounting. |
| 8 | **Diversity-collapse-as-overtraining** (Yuan?, arXiv:2606.15455 ⚠) + **pass@k inversion** (arXiv:2607.20543 ⚠) | "saturated problems contribute nothing" ≈ u_N(p)→0 at p→1; mean-up/pass@k-down reported | cite as convergent diagnoses; ours is the estimator-derived, *prescriptive* form (predicts where relabels are wasted before training). |
| 9 | **SPEED-RL** (arXiv:2506.09016 ⚠) | SNR-maximization theory for intermediate-difficulty selection | variance/SNR cousin of the identity; cite next to VIP in the allocation paragraph. |
| 10 | **GCL** (ICRA'25, arXiv:2409.19816) + **GACL** (IROS'25, arXiv:2508.02988) | curricula on BARN itself (ICRA track) | complementary objective (sim-to-real task-distribution grounding) vs ours (estimator-priced compute efficiency); use as baselines, not rivals — ideally combine. |

Also add (lower tier, one clause each): DUMP arXiv:2504.09710 (empirical
|advantage| as distribution-level learnability — nearest empirical use of our
exact quantity), PCL arXiv:2510.01135, gradient starvation arXiv:2605.07689 ⚠,
biased group-relative advantage arXiv:2601.08521 ⚠ (supports "evaluate with
the estimator"), RL-ZVP arXiv:2509.21880, pass@k-as-diagnostic
arXiv:2511.16231 (useful foil: we use pass@N as *utility*, not objective),
SimKO arXiv:2510.14807 / DPH-RL arXiv:2509.07430 (coverage-preserving
objective patches), HIR arXiv:2302.05206 (HER-for-LLMs progenitor — likely
missing from the bib), ECHO arXiv:2510.10304 (in-context relabeling analogue),
rollout-advantage replay arXiv:2606.04560 (estimator-aware, relabel-free
quadrant of the design space).

---

## 3. Gap analysis of the compact draft's related work

Already well covered: procurl, sfl, lilo, vcrl, plr, alpgmm, paired, accel,
greso, dps, vip, hora, vigor, dars, reinforceada, BBG, cai, rl2ml, yue, cui,
passk, her, gcsl, codeit, lfh (+ local body.tex additionally: dump, adarft,
sec, pkpo, scsdpo, agrae, goalgan, hgg, cher, skewfit, ziprl, agenther, hsl,
minimo, starcross, yuan).

**Missing and material (add before submission):**
1. arXiv:2607.00152 — the group-std identity. *The single most dangerous
   omission*: a reviewer who knows it will ask "isn't this known?"
2. MoPPS arXiv:2507.04632 — teacher-mechanism prior art. Omitting it looks
   like claiming the bandit as novel.
3. Actor-Curator arXiv:2602.20532 and/or Learning-Zone Energy
   arXiv:2605.17003 — the closed-form-learnability cluster.
4. F-GRPO arXiv:2602.06717 — concurrent N-dependent analysis.
5. arXiv:2606.15455 + arXiv:2607.20543 — strengthen the sharpening/coverage
   paragraph beyond yue/cui/yuan.
6. HIR arXiv:2302.05206 — the LLM instruction-relabeling progenitor; the
   hindsight paragraph currently starts the LLM story at CodeIt/LfH.

**Drop-in paragraph (compact draft, §Related work, after the ProCuRL/SFL/LILO
sentences):**

```latex
\paragraph{Concurrent exact analyses of group estimators.}
A 2026 cluster derives closed-form expected update magnitudes for
group-relative estimators: the group-standard-deviation identity
$\E\sqrt{k(G-k)}/G \to \sqrt{p(1-p)}$ with DAPO's discarded mass
$p^G+(1-p)^G$ \citep{groupstd}, an expected-absolute-advantage baseline
$2\sqrt{p(1-p)}$ inside a learned-curator bandit \citep{actorcurator}, and
variance- or SNR-based selection scores \citep{lze,speedrl}. These live on
the standard-deviation-normalized branch, whose expectation is flat in $N$.
The practical-MaxRL coefficient mass is instead exactly
$2(\passat{N}-\passat{1})$: unnormalized, genuinely $N$-dependent, and
interpretable as the estimator's own pass@$N$ gap --- the property our
sampler exploits and our $N$-sweep tests. MoPPS \citep{mopps} anticipates
our posterior machinery (streaming Beta posteriors with Thompson sampling
over prompts); we claim novelty only in the utility sampled toward, not the
bandit mechanism.
```

---

## 4. Polish: presentation upgrades for acceptance (compact ICLR draft)

The evidence is strong; the *packaging undersells it*. Ordered by leverage:

### 4.1 Abstract — cut the statistics, keep the ladder
Current abstract is ~340 words with 6 parenthesized statistical clauses,
including the paid-probe result and Holm p-values. Reviewers form the
accept/reject prior here. Recommend the `RESEARCH_REVIEW_HANDOFF_2026-08-11.md`
compact abstract as the base, with two edits:
- Drop the ProCuRL paid-probe sentence from the abstract entirely (keep in
  §Evidence + Limitations). In the abstract it reads as an unforced confession
  about an ill-posed comparison; in the body it is honest scope-mapping.
- End on the calibrated-but-affirmative line: "coefficient activity is a
  useful source of curriculum hypotheses, not a universal curriculum
  objective" — then the reporting recommendation. Keep at most TWO numbers
  in the abstract (Acrobot +.048; maze 6/6).

### 4.2 Contributions — rename the frames, not the claims
- C2 is currently titled "Controlled curriculum tests with mixed outcomes."
  "Mixed outcomes" is a self-review, not a contribution. Reframe:
  **"A positive--negative pair that maps the score's scope"** — the Acrobot
  win *and* the Digits refutation together are the contribution (a decision
  boundary for practitioners), not a win diluted by a loss.
- C3 "Qualified neural diagnostics" → **"Coverage accounting at scale"**:
  the mean-up/coverage-down split and the estimator-conditioned coverage
  ordering are findings; the qualifications belong in the sentences, not the
  headline.
- Add the reporting recommendation as an explicit artifact of the paper
  ("report mean@k beside pass@k computed from retained outcomes") — cheap,
  memorable, and it is what the field cites you for if the theory ages.

### 4.3 Preempt the three predictable reviewer questions, in-text
1. *"Isn't this the known group-std identity?"* → the drop-in paragraph
   above (§3).
2. *"Why is the Countdown result here if its metric is a proxy?"* → one
   sentence: "We include the aggregate because it motivated the
   preregistered raw-outcome protocol (E2c) that replaces it; we report the
   proxy's provenance rather than renaming it." (If E2c lands before the
   deadline, this section upgrades in place — the draft is already written
   to allow that.)
3. *"Two synthetic testbeds (Acrobot, Digits) — does anything transfer?"* →
   point at the maze wave as the real-gradient rung, and state the LLM
   results' role as scoped applied observations. One sentence in §Evidence
   intro tying the four families to the claim each supports.

### 4.4 Related-work ordering
Lead the section with the concurrent-2026 cluster (it's the novelty
defense), then heuristic curricula, then allocation, then coverage/hindsight.
Right now the section reads as a taxonomy; make the first paragraph do the
"here is the empty cell we occupy" work: *derived-from-deployed-estimator ×
N-aware × selection-not-objective* — every neighbor fails at least one.

### 4.5 Limitations — compress the artifact accounting
The registry/ledger paragraph (562 rows, 441 Acrobot, 5.08 GB payload…) is
appendix material; keep two sentences in §Limitations ("artifact coverage is
incomplete; App. X itemizes") and move the inventory. The Limitations section
should spend its budget on scientific scope, not bookkeeping.

---

## 5. ICRA-track positioning (for the paper in `claude-fable-plan.md`)

**Related-work skeleton (6-page budget: ~0.5 page):**
1. *Curricula in robot RL:* ALP-GMM, TSCL, GoalGAN, Setter-Solver, self-paced
   (Klink), CURROT, PLR/ACCEL/PAIRED — one clause each on the criterion
   (learning progress / regret / bands / OT).
2. *Success-band practice:* terrain promotion (Rudin arXiv:2109.11978), Lee
   Sci-Robotics 2020, Margolis RSS'22, ADR (arXiv:1910.07113) — "the field
   already targets intermediate success rates with hand-set thresholds; we
   derive the target and its budget dependence."
3. *Learnability:* SFL, ProCuRL (+ LILO for LLM port) — the p(1−p) family;
   the SFL finding that PLR's regret proxies collapse to success-rate
   statistics is an argumentative asset: cite it as *why success-rate-based
   utility is the right primitive to make principled*.
4. *BARN:* benchmark (arXiv:2008.13315), challenge reports, **GCL
   (arXiv:2409.19816) and GACL (arXiv:2508.02988) as the domain SOTA and
   baselines.** Frame as complementary: GCL/GACL ground the task distribution
   in real-world relevance; u_N prices tasks by training value at the current
   policy and budget. If feasible, run u_N *inside* the GCL/GACL loop as an
   arm — "improves the lab's own SOTA" is the strongest possible ICRA result
   and inoculates the "how does this relate to GCL?" review.

**Baselines to implement:** uniform, p(1−p) (SFL-faithful), hand
promotion/demotion (Rudin-style), GCL/GACL (existing lab code), + N-sweep
ablation. This matches the matrix in `claude-fable-plan.md` §2.2 — replace
the generic "PLR if time" slot with GCL/GACL.

**ICRA framing note:** lead with the success-band *practice* (every legged
and navigation pipeline hand-tunes promotion thresholds), then "here is the
threshold-free, budget-aware score those rules approximate, derived from the
estimator you already run."

---

## 6. Action checklist

- [x] **2026-08-11:** Verified all 10 flagged arXiv IDs — every ID resolves
      and matches. Full-text deep dive on the three biggest threats:
      2607.00152 derives only std-normalized quantities (no pass@N identity);
      2602.20532's `2√(p(1−p))` is N-flat and only a baseline; MoPPS
      Thompson-samples toward `|p̂−0.5|` (KDD 2026). **Novelty wedge
      confirmed.** Remaining: full-text read of 2607.09042 (LfH) before
      submission.
- [x] **2026-08-11:** Compact draft extracted from release branch
      `9277141` into the worktree as `paper/body_iclr.tex` (+
      `iclr2027_conference.sty`, wrapper `main_iclr2027.tex`) and polished:
      abstract rewritten per §4.1 (ProCuRL moved out, two headline numbers);
      contributions retitled per §4.2 + reporting recommendation added;
      evidence roadmap paragraph + Countdown preemption sentence added per
      §4.3; limitations artifact paragraph compressed per §4.5; related work
      opens with the empty-cell framing + new "Concurrent exact analyses"
      paragraph; 6 bib entries added (groupstd, actorcurator, mopps, lze,
      passkinv, hir). Terminology scan clean; braces balanced. **Not yet
      compiled — no TeX engine on this machine; page count must be verified
      in the release's pinned environment during reconciliation.**
- [x] **2026-08-11:** Website (`docs/index.html`): hero date bumped, lede
      tightened (disclosures live in the scoreboard card), stale §8 abstract
      card (pre-retraction, "provably yields the ML gradient" overclaim)
      replaced with the calibrated compact abstract, PAPER.md pointer
      replaced with body_iclr.tex/body.tex roles.
- [ ] ICRA draft: related-work skeleton per §5; decide whether GCL/GACL
      integration arm is feasible by the Aug 24 checkpoint.
- [ ] After reconciliation compiles, run `/ars-reviewer` on the revised
      compact draft to check the reframing reads as calibrated rather than
      defensive, and confirm the conclusion still ends on page 9.
