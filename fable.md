# Fable review — research, experiment, and ICLR-draft status (2026-08-26)

**Scope.** Read: `AGENTS.md`, `READINESS.md`, `SUBMISSION_GAP.md`, `SCHEDULE.md`,
`NEXT_EXPERIMENTS.md`, `LLM_PERIMETER_DECISION_2026-08-26.md`,
`CRITICAL_REVIEW_LATEST_DRAFT_2026-08-20.md`, `hopper/HOPPER_STATUS.md`, the
uncommitted `LITERATURE_POSITIONING.md` diff, `granularity_flip/GROUP_LAW_FLIP_PREREG.md`,
`paper/CLAIM_TRACE_ICLR.md`, the full main text of `paper/body_iclr.tex`, and the
built `paper/main_iclr.pdf` (18 pp; main text ends on p. 9, references start p. 10).
I also ran the **marker-only** Hopper status commands from `HOPPER_STATUS.md`.
No result JSONL, log, telemetry, or analyzer was opened. This file is advisory,
not an evidence source (AGENTS.md §2.6).

**Closure update:** P0 was unblinded under the frozen single-use path later on
2026-08-26 and returned `supported`; §8 is the current addendum. Sections 0–7
preserve the outcome-blind review state that preceded that action.

---

## 0. The one thing that matters today

**P0 (count-law flip, `group-law-flip-v1-20260820-001`) is terminal and unclosed.**

| check (marker-only, 2026-08-26) | value |
|---|---|
| Slurm accounting, `group-law-flip-v1` since 08-21 | 40 `COMPLETED 0:0`, plus 8 in the first chunk → 48 |
| `campaign-status`: final blocks / COMPLETE / SHA256 manifests / arm receipts | 48 / 48 / 48 / 96 |
| invalid final blocks / incomplete quarantines | 0 / 0 |
| structural state | `ALL_COMPLETION_MARKERS_PRESENT` |
| live queue for `lwang44` | empty |
| local canonical copy `/data/robotixx/group_law_flip/...` | **does not exist** |
| `GROUP_LAW_FLIP_ANALYSIS.json` | **does not exist** |

The last allocation ended 2026-08-21 ~08:29. The campaign has been sitting
complete for five days while the draft, `AGENTS.md`, and `HOPPER_STATUS.md` all
still say "running blind." `hopper/close_group_law_flip.sh` and
`hopper/watch_and_close_group_law_flip.sh` are written but **uncommitted**, and
the hash-pinned closure has never been invoked.

This is the single decisive experiment for the paper's spine ("score the count
law rather than the mean pass rate"). Every editorial question below is
downstream of its verdict. **Action 1: commit the closure scripts, run
`close_group_law_flip.sh` once (it is single-use by construction), and write the
registered verdict — whichever branch — the same day.** I did not run it: the
analyzer is single-use and the PI should own the unblinding.

Second unclosed terminal campaign: the AMaze gate confirmatory rerun
(`/data/robotixx/ued_bench/gate-confirmatory-20260819`, 20/20 cells) — still no
`DONE` markers / `AMAZE_GATE_ANALYSIS.json`. Lower stakes, but it is also
evidence you paid for and have not read.

Third dated item: the LLM go/no-go was due **today**. The pre-commitment
recommends (b) de-scope unless a cheap frozen-checkpoint smoke shows the
coarse-posterior treatment moves the dead-prompt fraction. Nothing in the tree
shows that smoke was run. Record the decision explicitly (a one-paragraph
dated addendum to `LLM_PERIMETER_DECISION_2026-08-26.md`). Default is (b).

---

## 1. Where the science actually stands

**Tier 1 (proved, machine-checked) — solid and genuinely nice.**
- `A_N(Q) = 2(q_N − p̄)` for any joint binary law; the count-law functional
  `A_E(z) = Σ_k P(K=k|z) M_E(k)` with closed-form masses for MaxRL/RLOO/GRPO.
- `u_N(p) = p(1−p)·w_{N−1}(p)`: the $p(1-p)$ literature is the $N=2$ slice.
- `T = N−1` truncation for the drop-all-fail centered convention.
- Cor. 2 (granularity): plug-in over-predicts by exactly `2[Pr(K=0|z) − (1−p̄)^N]`
  in the mixture-of-i.i.d.-tasks regime.
- New, uncommitted: the relabel-degeneracy lemma (`test_relabel_degeneracy.py`) —
  per-member relabeling of an all-fail group lands at `k=N`, which carries
  **zero** mass for every permutation-equivariant estimator. This is a small but
  sharp result and the only admissible repair (group-consistent relabel) is
  stated with its own weakness. Worth two sentences in the paper (see §4).

**Tier 2 (registered, confirmed).** Acrobot `u_16 − p(1−p) = +.0480 [+.0209,+.0738]`,
20 paired seeds, plus two same-seed portability replications (`+.0322`, `+.0307`).
Also: MAZE-SCORE secondary `u_32 − uniform = +.0089`, and maze wave-2 estimator
ordering 6/6 per sampler.

**Tier 3 (registered, bounded/refuted).** Peak-location specificity (curve rises
past deployed N, argmax at `u_64`); AMaze standalone priority (starved);
MAZE-SCORE primary `u_32 − p(1−p) = −.0032 [−.0054,−.0011]` practically ruled
out; Digits estimator×sampler interaction unsupported; Countdown corrected-gate
dose (P-R1) refuted; GSM8K delivery gate missed by .00148.

**Tier 4 (open).** P0 (terminal, unread); AMaze gate rerun (terminal, unread);
LLM interaction (1-of-2 seeds).

**Honest summary of the evidence base.** One 640-parameter registered positive
with two portability copies, one 1.26M-parameter registered *negative* that the
theory retroactively rationalizes, and one preregistered causal test of that
rationalization which is finished and unread. The paper is currently a
theory-plus-boundary-map paper whose central empirical claim is being carried by
an Acrobot actor with 640 parameters. That is the vulnerability reviewers will
name first, and P0 is the only thing in the pipeline that can change it.

---

## 2. Draft assessment (`paper/body_iclr.tex`, build of 2026-08-20)

### What is strong
- Fig. 1 (one mean pass rate, two worlds) is an excellent opening — exact,
  visual, and it *is* the paper. Keep it as the first figure.
- Contribution list tiered by entitlement is unusual and reads as credible.
- Prop. 1 + Cor. 1 + Cor. 2 are clean; the SFL-is-RLOO's-count-law identity
  (already in Related Work) is the right way to position, and it turns a
  potential reviewer "gotcha" into a contribution.
- Numbers are traced (62 rows, all TRACED). Main text fits 9 pages.
- Limitations are real limitations, not disguised strengths.

### What a strong ICLR reviewer will write (ranked)

**R1. "The main empirical result is a 640-parameter Acrobot actor."**
Registered, yes; but the paper's audience (RLVR for LLMs) will read Acrobot,
Digits bandit, and a 36-task CPU chain as toys, and the one neural-scale study
is a negative. The abstract's `+.0480` is on Acrobot. Without P0 landing
positive, the "score the count law" prescription in the takeaway box
(item ii) is an *untested* recommendation in a paper whose whole ethic is not
recommending untested things. **This is the paper's central risk and P0 is
the fix.**

**R2. "Cor. 2 predicts a calibration bias, not a sign; the MAZE-SCORE story is
post hoc."** The draft says this itself, repeatedly and correctly — but says it
so many times ("consistent with, not a prediction of"; "post-hoc; no registered
quantity changes"; "not sole mediation") that §3.4 reads as defensive. Once P0
is in, say it once, in the P0 paragraph, and delete the other three hedges.

**R3. "What is the practical algorithm?"** FrontierMax is described in one
paragraph (`sec:method`) with the concrete estimator of `q̂_z − p̄_z` only in
the takeaway box. There is no algorithm box in the main text (Alg. is in
App. B). A reader who wants to *use* the result needs: the sufficient-statistic
posterior `(W, Z, S, S2)`, the score for each estimator, the floor, and the
sampler — in ≤10 lines. That is what P0 tests, so it should be visible.

**R4. "The peak-specificity negative undercuts the theory."** §3.2 shows
performance rises *past* deployed N and argmax is `u_64`, and the branching
analysis shows `U_H` peaks at `p ≈ .02`, far below `p*_16`. The paper's reply —
"the shape helps, the location does not; deployed N is a floor" — is honest but
leaves the reader asking *why the theory's peak is wrong*. Offer the mechanism
you already have: activity is a per-update envelope, downstream utility
compounds over H updates, and compounding favors harder tasks (the chain
result). One sentence, plus a pointer to the transfer-matrix result. Don't leave
this as an unexplained refutation of your own derived quantity.

**R5. "Digits is a mess."** MaxRL favors `u_8` by `+.208` (23/24), RLOO also
favors it by `+.177`, *both* fall below uniform. That is: the estimator-
specific prediction failed, and the curriculum hurt. It currently sits under
"Exact-probability counter-test" inside the theory section (!), and the
abstract counts Digits as one of the "two further platforms" only implicitly.
Either move Digits to §3 evidence with its negative fully owned, or move it to
the appendix and stop letting it support the "replicates" language.
Actually check: the abstract says "replicates on two further platforms" — those
are the two same-seed Acrobot campaigns, which the paper itself says are a
*portability check, not additional independent seeds*. A reviewer will call
that "replicates" wording overclaiming. Use "reproduces on two further
platforms with the same seeds" or drop the clause.

**R6. Related work.** Good coverage, but two gaps a reviewer in this subfield
will hit: (a) the SFL variable-N shrinkage defect (`n/(n+1)` vs `n/(n−1)`) in
the uncommitted `LITERATURE_POSITIONING.md` is citable and sharpens the wedge —
one sentence; (b) SPEED-RL (SNR-maximization) is in your positioning doc but
absent from the bib. Also the bib has 45 entries; ICLR reviewers expect the
2025–26 RLVR-curriculum canon (DUMP, SEC, LILO, MoPPS, VCRL, GRESO, DPS, VIP,
HORA are there — good).

**R7. Provenance.** Acrobot/Digits locks are internal hashes, no public
pre-execution commit; Digits replay 5 GB has no URI; maze checkpoints external.
The draft discloses all of it, which is right. Mint one OSF/Zenodo DOI for the
hash bundle before the deadline and cite it in App. D; this converts "internal
hash" into "timestamped hash" for ~1 hour of work.

**R8. Minor but visible.**
- Fig. 2 (claim scoreboard) caption is denser than the figure; readers will not
  parse "rows there are different endpoints on their own studies' scales."
- "Coefficient activity" vs "advantage mass" — you gloss once, good; but
  `A_N`, `u_N`, `𝒜_{E,N}`, and `M_E` are four symbols for two ideas. Consider
  `u_N` only in prose, `𝒜` only in Def. 1.
- The takeaway box item (iv) (relabel + gate) refers to results that are
  appendix-only. Either promote the gate result (10/10 seeds, 98% recovery) to
  the main text or drop item (iv).
- Abstract has six numbers; AGENTS.md P1 says four.
- `OPENREVIEW_ABSTRACT_CANDIDATE.md` still describes the pre-pivot paper.
- `GRPO` mass formula in Def. 1 uses sample-SD; say `(N−1)` denominator
  explicitly once, since the group-std identity paper (`groupstd`) uses the
  other convention.

---

## 3. Experiment advice — what to do with the ~3 weeks left (abstract 09-18, paper 09-25)

Ordered. Do not start item k+1 while k is red.

### E1. Close P0 (today). Both branches are pre-drafted; execute the frozen one.
- **Supported (`grouplaw − plugin ≥ +.005`, rule passes):** this is the paper's
  headline. Restructure: abstract leads with "scoring the count law recovers X
  of the loss on the same substrate where the plug-in lost"; §3.4 becomes
  "diagnosis → intervention → result"; takeaway (ii) is now tested; Fig. 2B gets
  a fourth row (filled marker). Contribution 4 moves from "open" to
  "registered and confirmed." The paper then has *one* neural-scale positive
  causal intervention, which changes its category from boundary map to method
  paper with a boundary map.
- **Practically ruled out / inconclusive:** report it as a fourth boundary.
  The theory is unharmed (Cor. 2 predicts calibration, not endpoint) and the
  paper's thesis sharpens to "even correct coefficient activity is not
  learning utility" — which is already the last sentence of the AGENTS.md
  spine. Do **not** rerun, add seeds, or reopen the MAZE-SCORE diagnosis.
  Remove takeaway (ii)'s prescription or mark it "untested."
- **Treatment not delivered:** the replay-delivery artifact
  (`GROUP_LAW_FLIP_DELIVERY_REPLAY_2026-08-20.json`) already exists; report the
  delivery statistic and say so.

### E2. Close the AMaze gate rerun (this week). Run the outcome-blind checkpoint-
budget closure, then the frozen analyzer once. If it confirms the 5-seed
development result (`gate ≈ .590`), §3.3 gains a registered number and the
takeaway (iii) "gate on a richer signal" becomes Tier 2. If not, §3.3 stays a
development negative — cheap either way.

### E3. LLM decision (today): take (b). Record it. Do not spend 40–80 A10G-h on
a gate that already failed once unless the frozen-checkpoint smoke is run
first *and* moves the dead-prompt fraction. The paper is self-contained without
it. If you have idle GPU, the higher-value LLM item is not RL training at all —
it is E4.

### E4. Frozen-checkpoint LLM count-law calibration (optional, ~4–8 A10G-h,
no training). This is the cheapest way to answer R1 without a training run:
take one frozen SmolLM/Qwen-0.5B checkpoint, sample N=16 on GSM8K or Countdown
grouped by the *coarse* unit practitioners actually use (difficulty bucket /
template / source dataset), and measure `Pr(K=0|z)` vs `(1−p̄_z)^N` per bucket.
Cor. 2 says the gap is `≥ 0` and the paper says it was 2.2% vs 51.2% on mazes.
If the same gap is large on an LLM pool, you have an LLM-scale *measurement*
of the phenomenon — Tier 2' descriptive, one figure, no training, no delivery
gate. That directly answers "does this matter at LLM scale?" with a number.
Preregister the bucket definition and the summary statistic; this is a
measurement, so the rule can be simple ("report gap per bucket; no
hypothesis test").

### E5. Do not start: new substrates, reasoning-gym, K&K/DUMP head-to-head,
Cosmos/LIBERO, BARN. `NEXT_EXPERIMENTS.md` and `READINESS.md` are pre-pivot
roadmaps; treat them as archived. Nothing in them can land at Tier 2 before
09-12.

---

## 4. Manuscript advice — concrete edits, ordered

1. **After P0**: rewrite abstract (≤4 numbers: Acrobot `+.0480`, MAZE-SCORE
   `−.0032`, P0 verdict number, and the CI of whichever is headline).
   Regenerate `OPENREVIEW_ABSTRACT_CANDIDATE.md` and settle the title with
   the PI (`SCHEDULE.md` says the title was conditional on P-F2; the current
   spine no longer matches "The Estimator Decides").
2. **Add a 10-line Algorithm box to the main text** (count-law posterior
   `(W,Z,S,S2)`, score per estimator, floor, categorical sampler). Cut
   ~0.3 page from §3.2's secondary axes paragraph ("2.40 fewer groups…") to pay
   for it.
3. **§2 "Exact-probability counter-test" (Digits)** → move into §3 as its own
   short subsection *or* to App. C; stop counting it as replication. Fix
   "replicates on two further platforms" → "reproduces on two further
   platforms with shared seeds."
4. **Collapse the Cor. 2 hedges** to one sentence in §3.4; the rest is
   repetition once P0 exists.
5. **Add the relabel-degeneracy lemma** (two sentences in §2 after the dead-zone
   paragraph, proof in App. A, tests already written). It explains why
   takeaway (iv) says "group-consistent relabel" and forecloses the obvious
   reviewer suggestion.
6. **Answer R4** with one mechanism sentence on compounding + pointer to the
   transfer-matrix / `U_H` result.
7. **Related work**: commit the SFL correction (it is already in the paper's
   prose — confirm every occurrence of "heuristic" is gone in README/site
   too); add SFL variable-N shrinkage sentence; add SPEED-RL.
8. **Provenance**: OSF/Zenodo DOI for `manifest.json` + Acrobot lock + Digits
   descriptors; cite in App. D and Reproducibility.
9. **P2 hygiene** from AGENTS.md: rebuild `paper/results/manifest.json` around
   the figures actually used (`fig_counterexample`, `fig_claimmap`,
   `fig_mazescore`, `fig_regimes` are not in the old manifest); fix
   `reproduce.sh` (byte mismatch at `fig1_utility.pdf`, `../maxrl` escape);
   clean anonymous clone build; identity-leak scan of PDF metadata
   (`pdfinfo` currently shows nothing suspicious but check `\pdfinfo` author
   fields and figure PDFs' Creator strings).
10. Refresh `CLAIM_TRACE_ICLR.md` with one unambiguous total after edits.

---

## 5. Calendar check

| date | milestone | status |
|---|---|---|
| 08-26 (today) | LLM go/no-go | **bound: (b) de-scope** (see §7) |
| 08-26 | P0 closure | **supported; integrated at Tier 2** (see §8) |
| 09-12 | freeze claim table / perimeter | reachable only if P0 + AMaze close this week |
| 09-16 | OpenReview title + abstract | depends on P0 branch |
| 09-18 AOE | abstract deadline | — |
| 09-22 | freeze draft | — |
| 09-25 AOE | paper + supplement | — |

---

## 6. Bottom line

The theory is done and correct; the discipline is exemplary; the writing is
9 pages and traced. What separates this from a solid accept is not more
experiments — it is *reading the one you already ran*. P0's verdict decides
whether the paper is "a method with a priced boundary" or "a boundary map
with a method hypothesis," and both are publishable if written to their tier.
Close P0 and the AMaze gate this week, take (b) on the LLM run, spend any
spare GPU on the frozen-checkpoint calibration measurement (E4), and put the
remaining time into the algorithm box, the Digits relocation, and provenance.

---

## 7. Update 2026-08-26 (later) — review of Codex's pass

Checked `git status`/`git diff` against §0 and §4 above. Nothing is committed
yet (HEAD is still `b3d90b8`, 2026-08-20); all of the below is working tree.

### Verified as done and correct
| item | verdict |
|---|---|
| `hopper/close_group_law_flip.sh --preflight-only` | ✅ Conjunctive gate: 0 live, 48 completed, 0 failed/nonterminal, helper `COMPLETED\|0:0`, 48/48/48/96 markers, 0 invalid, 0 quarantines → else `die`. Analyzer hash pinned; refuses to run if canonical local copy or analysis JSON already exists (single-use). `bash -n` OK. `fetch_group_law_flip.sh` exists. Post-analysis assertions check schema/protocol/campaign/48 blocks/decision ∈ frozen set. This is the right shape. |
| P0 status wording (paper ×3, README, site, `HOPPER_STATUS.md`, `AGENTS.md`) | ✅ "terminal but still sealed pending the single-use analysis" — accurate and tier-correct. Contribution 4 still "named as open." |
| LLM go/no-go | ✅ Option (b) bound with a dated decision record; states explicitly that it creates no evidence and doesn't reinterpret the .00148 miss. `AGENTS.md` P3 marked done. |
| SFL / replication wording | ✅ Abstract: "reproduces on two further platforms using the same seeds." Related work: "recovers SFL as the count-law curriculum for RLOO rather than replacing it as a heuristic" + variable-N clause. Site/README "heuristic" residuals are all in the corrected sense. |
| Claim-trace total | ✅ One unambiguous line: 82/82 (58 base + 24 addendum). |
| `OPENREVIEW_ABSTRACT_CANDIDATE.md` | ✅ Regenerated verbatim from the new abstract. |
| Terminology | ✅ "advantage mass" → "coefficient activity" swept through README/site with one gloss retained. |

### Issues to fix (small)
1. **Abstract number trim is asymmetric.** `+.0480 [+.0209,+.0738]` keeps its CI; `−.0032` lost its `[−.0054,−.0011]`. A reviewer reads a bare point estimate as "unqualified." Either treat an interval as *one* number-object (then the abstract has 2 objects and both CIs fit within the four-number rule) or drop both CIs. Recommend: restore the MAZE-SCORE CI; the rule's intent is "don't pile up numbers," not "strip uncertainty."
2. **PDF not rebuilt.** `paper/main_iclr.pdf` is still the Aug-20 build; the tex edits are unrendered. Rebuild and re-check the 9-page boundary (the P0 sentence in §3 grew by one line).
3. **Nothing is committed.** The closure scripts are still untracked. Commit the whole working tree now (status wording + decision record + closure tooling) as one "P0 preflight green; LLM de-scoped" commit *before* running the closure, so the pre-unblinding state is in history.
4. `LLM_PERIMETER_DECISION` removed "plus the creation channel" from the perimeter sentence. That matches the current paper (gate result is appendix-only), but make sure `AGENTS.md` §1 spine and takeaway (iv) agree — takeaway (iv) still prescribes relabel+gate from an appendix-only result (§2 R8 above).

### Not yet done from the earlier list
- §4 items 2 (algorithm box), 3 (Digits relocation), 4 (collapse Cor. 2 hedges), 5 (relabel-degeneracy lemma — tests still untracked), 6 (R4 mechanism sentence), 8 (DOI), 9 (manifest/reproduce.sh/anon clone), and E2 (AMaze gate closure) are untouched. All are still appropriate; none should wait on P0 except 3/4 wording.

### Guidance — next 48 hours, in order
1. **Commit** the working tree (see issue 3).
2. **Run `hopper/close_group_law_flip.sh`** (not preflight-only). PI-owned action. Read only `GROUP_LAW_FLIP_ANALYSIS.json`'s `primary_grouplaw_minus_plugin.decision` first, then the frozen branch text from `GROUP_LAW_FLIP_PREREG.md` §verdict. Write `granularity_flip/GROUP_LAW_FLIP_RESULT_2026-08-2x.md` in the same form as `hopper/MAZE_SCORE_RESULT_2026-08-18.md`. Commit analysis JSON + result memo together.
3. **Propagate the verdict** per §3 E1 branches: contribution 4, abstract (now the P0 number is the 3rd/4th abstract number), §3.4, Fig. 2B row, takeaway (ii), limitations, claim trace (row 83+), registry, README, site. Do this in one commit so the paper never carries a half-propagated P0.
4. **Close the AMaze gate rerun** (E2) — same discipline, lower stakes.
5. Then the P0-independent manuscript work: algorithm box, Digits relocation, relabel lemma + commit its tests, R4 sentence, DOI, manifest/reproduce.sh, anonymous-clone build.
6. Rebuild PDF after every tex edit; the 9-page limit is currently at the edge.

Calendar unchanged: perimeter freeze 09-12, title/abstract lock 09-16, abstract 09-18 AOE, paper 09-25 AOE.

---

## 8. Update 2026-08-26 (closure) — P0 supported

The complete pre-unblinding tree was committed at `8349888`. The single-use
closure then rechecked the 48-block terminal/marker conjunction, verified every
remote block hash, retrieved the canonical campaign, passed the frozen 2×48
structural validator, and invoked the hash-matched analyzer once.

Frozen primary `grouplaw - plugin`:

| quantity | result |
|---|---:|
| cov-AUC difference | **+.00666** |
| paired-bootstrap 95% CI | **[+.00441,+.00887]** |
| exact paired sign-flip p | **9.56e-7** |
| positive / negative / tied blocks | **40 / 8 / 0** |
| mean visit TV | **.33597** (gate ≥.05 passed) |
| frozen verdict | **supported** |

This is Tier 2: the count-law correction is causally relevant on the
MAZE-SCORE substrate. It does not show that Corollary 2 predicted the learning
sign, that the correction alone explains the earlier contrast, or that either
arm beats `p(1-p)`. The result artifact is
`curriculum_maxrl/group_law_flip/GROUP_LAW_FLIP_ANALYSIS.json`; the calibrated
memo is `granularity_flip/GROUP_LAW_FLIP_RESULT_2026-08-26.md`.

The central empirical vulnerability in §2 R1 is therefore materially reduced:
the paper now has a preregistered neural-scale causal intervention in addition
to the Acrobot score-shape positive and the registered boundaries. The next
red item is the outcome-blind AMaze gate closure, followed by the independent
manuscript and artifact queue already listed in §7.
