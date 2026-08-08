# Final Review — Response, Verification, and Guidance

**Date:** 2026-08-07
**Responding to:** `FINAL_ICLR_REVIEW_AND_COMPLETION_GUIDE_2026-08-07.md` (commit `221bfb6`)
**Repository state:** post-`221bfb6`, `reproduce.sh` green as of 2026-08-07
**Method:** every checkable claim in the codex review was verified against raw
artifacts, run logs, and the live ICLR 2027 pages before agreeing or
disagreeing. Recomputations are noted inline.

---

## Part I — Verification of the codex review

### I.1 Claims verified TRUE (recomputed or checked at source)

| Review claim | Verification |
|---|---|
| Wave-2 block-averaged covAUC contrasts: +0.02264, +0.01883, +0.01963, +0.01903, +0.00661, +0.03025; mean +0.01950; 95% t CI [+0.01148, +0.02752] | Recomputed from `curriculum_maxrl/maze_gpu_factorial/results_factorial_wave2.json`. **Exact match.** |
| Wave-2 easy-band block contrasts: 4 positive, 1 exact tie (seed 7), 1 negative (seed 10); mean +0.08333; CI [−0.00330, +0.16996] — includes zero | Recomputed from the same artifact. **Exact match.** |
| Cross-wave block-averaged covAUC positive 12/12; mean +0.02175; CI [+0.01663, +0.02688] | Recomputed from wave-1 + wave-2 artifacts. **Exact match.** |
| The P-F3 10/12 count treats 12 sampler contrasts as independent when they share 6 seed blocks | True at the artifact level: `WAVE2_VERDICT.md` quotes sign-test p = 0.039 over 12 pairs. The correlated-unit criticism is valid. |
| g3p failed its preregistered treatment-delivery gate: min 0.413 (< 0.50, passes), run-mean 0.601480 (fails < 0.60 by 0.00148); final mean@4 ≈ .105, pass@4 ≈ .198 | Recomputed from `/tmp/h6_cell_g3p.log` (50 steps of `train_all_datasets_binning/fraction_of_prompts_in_[0.0, 0.0]`). **Exact match.** Gate thresholds confirmed in `smollm/run_h6_steered.sh` (P-S1). |
| Verdict artifact stale | Confirmed: `e_llm1b_verdicts.json` marks g3p `FAIL/inconclusive-by-design` but carries `final_pass4: null`, and the cell JSON has only the step-0 val row. It also embeds absolute `/home/ec2-user/...` paths. |
| `fig9_passk.py` hard-codes per-k arrays from one seed | Confirmed: `K`, `B1`, `B2`, ... arrays in-script; `manifest.json` honestly lists `"inputs": []` with a transcription note — but `body.tex:1838` says "none are hard-coded in figure scripts," which is false for this figure. |
| SFT warmstart overlaps 27/128 (21%) of tier-0 eval; tiers 1–2 clean | Confirmed at `../maxrl/curriculum_maxrl/VERL_AUDIT.md` F12. `data_integrity_check.json` checks only RL train/test overlap; `body.tex:1868-1870` ("zero train/eval task overlap") is true for the RL split but silent on SFT. |
| LLM replay arm is higher-dose, not dose-matched (`ppo_epochs=2` ≈ 2× all live groups vs ~19% added relabel groups) | Confirmed; §6.8 itself discloses this ("bounds what extra updates buy rather than isolating the direction term"), but the conclusion (`body.tex:1346`) calls it a "dose-matched replay control" — mislabeled there. |
| Moderate gate point ran under-gated (decay bug); corrected strong gate failed (P-R1 refuted 3/3) | Confirmed: disclosure at `body.tex:1129-1132`; ARM-A verdict in `countdown_reviewer_arms/`. Abstract still says "one validated operating point" (`body.tex:19`). |
| Main text ~17 pages; ICLR 2026 style; no AI-use statement | Confirmed: references heading on p. 18 of the 24-page `main_iclr.pdf`; `main_iclr.tex` loads `iclr2026_conference`; only a Reproducibility section exists. |
| ICLR 2027: abstract 2026-09-18 AOE, full paper 2026-09-25 AOE, 9-page strict initial limit, **required** AI-use statement, recommended reproducibility statement | Confirmed live at iclr.cc/Conferences/2027 (CallForPapers + AuthorGuidelines) on 2026-08-07. |
| Word-boundary relabel rewrite can corrupt decimals/intermediate arithmetic | Confirmed: `verl_integration/vendored/hindsight.py:173` uses `\b<old_target>\b`; `\b12\b` matches the "12" in "12.5" and in "4 * 3 = 12". The failure modes are real. |

The codex review's numerical work is accurate everywhere I recomputed it. That
is worth saying plainly: this is a review whose arithmetic can be trusted.

### I.2 Corrections to the codex review

1. **The invalid p=.039 is not in the manuscript.** `body.tex` cites P-F3 as
   "10/12 pairs" with no p-value; p = 0.039 appears only in
   `WAVE2_VERDICT.md` and the prereg. The exact-language replacement in
   review §10 targets a sentence that does not exist in `body.tex`. The
   *substantive* fix (block-level reporting, no independence claim over
   sampler pairs) still applies — to the artifact docs and to every "10/12"
   and "24/24" phrasing in body.tex (lines 29, 164, 791, 803, 840, 894-895,
   1342, 1727-1729).

2. **The review misses the newest result with the same defect it corrects.**
   The premium reanalysis (commit `2070ad9`, §6.3b, `premium_reanalysis.json`)
   quotes "positive in 24/24 paired blocks both waves" — the identical
   correlated-pair counting E1 is meant to eliminate. E1's fix must cover it:
   restate at block level (n = 12) or drop the count and keep the direction.

3. **The manuscript is more honest than the review's ledger implies in two
   places.** §6.8 already states the dose caveat in full and draws only the
   bounding conclusion; §6.9 already discloses the decay bug and calls the
   deployed point "under-gated." The residual problems are the *abstract*
   ("one validated operating point"), the *conclusion* ("dose-matched"), and
   the claims table — i.e., propagation failures, not concealment. This
   matters for effort estimation: these are one-line edits, not rewrites.

---

## Part II — Where I agree and where I disagree

### II.1 Agree (adopt as-is)

- **The central thesis and three-contribution structure** (review §2, §8).
  Exact estimator result + balanced maze confirmation + Countdown sharpening
  is the right paper. The evidence supports "the estimator conditions what
  curricula and recycling can do, and mean accuracy hides a coverage cost" —
  and does not support more.
- **E1 (block-level factorial reanalysis)** — analysis-only, mandatory. The
  primary claim *survives* the correction (12/12 blocks, CI well above zero),
  so this costs nothing but honesty.
- **GSM8K to appendix as a treatment-delivery negative** (review §5). Both
  g3s and g3p failed the P-S1 telemetry gates; the pre-committed branch in
  `run_h6_steered.sh` (lines 86–88) says exactly what to conclude: prompt-level
  steering at this scale/budget did not deliver the treatment. Execute that
  branch. Do not relaunch a full cell without a cheap steering pilot that
  passes the gate first.
- **E3 (multi-seed pass@k curve)** and **E4 minimum repair (tier-0 on the 101
  clean tasks + disclosure)** — both cheap, both close real objections.
  Between-arm contrasts survive E4 because all arms share the SFT checkpoint;
  only absolute tier-0 numbers are tainted.
- **E5 decision rule for the gate**: corrected-code replication or demotion
  from abstract/contributions to appendix heuristic. The abstract's
  "validated operating point" is currently wrong under the paper's own
  standard — the point that was validated is not the mechanism the paper
  describes.
- **E6 (rewrite safety)**: the decimal/intermediate-equation corruption is
  real and the fuzz-test prescription is right. This is a release blocker for
  the artifact even if results don't change (corrupted rewrites are ~rare on
  this pool but the code ships).
- **All ICLR 2027 compliance items** (§12) — verified correct, including the
  mandatory AI-use statement, which for this project must disclose the
  documented AI-assisted workflow. Non-negotiable and easy.
- **The nine-page blueprint and float budget** (§8) — the "escalating ladder"
  organization is the single largest presentation risk. Organize by argument;
  the chronology of retractions moves to a scope-and-falsification paragraph
  plus appendix. Target 8.5 pages.
- **The schedule** (§13), including the 2026-08-28 hard stop on new training
  and abstract-identity freeze at 09-18.

### II.2 Disagree or amend

**D1. P-F3 should be reported as "met its registered bar," not relabeled
"descriptive."**
The review asks that the easy-band result be demoted to descriptive wholesale.
But P-F3 was preregistered *at the pair level* with a bar of ≥7/12 and landed
10/12. Retroactively re-unitizing a registered secondary because its unit now
looks unflattering is the same move — in the safe direction — as pooling
correlated pairs was in the flattering direction. The discipline this project
has been scrupulous about cuts both ways. The right reporting:

> "P-F3 met its preregistered bar (10/12 pair-level contrasts, bar ≥7/12). We
> note the registered unit counts two correlated sampler contrasts per seed
> block; at the block level the contrast is positive in 4/6 with one tie and
> one negative, and the interval includes zero — so we treat *localization* to
> the easy band as suggestive rather than established."

No p-value, no "confirmed" without the unit caveat, but also no erasure of
what was registered and met. Same treatment for wave-1's exploratory 9/12.

**D2. E2 (true dose-matched replay) is strongly recommended, not a submission
blocker.**
Three reasons. (a) A genuinely matched replay control already exists at the
CPU rung: the §6.1 battery's live-gradient replay is slot-matched, captures
83% of recycling's AUC, and isolates a +0.037 direction term at 5 seeds —
the causal decomposition the review wants exists, one rung down. (b) The LLM
arm's §6.8 text already claims only the bound. (c) The applied headline
(sharpening: mean up, pass@k down, 3/3 seeds) does not depend on the replay
arm at all. What *is* required for submission: fix the conclusion's
"dose-matched" label (→ "higher-dose live-group replay control") and soften
the abstract's "not relabel-specific" to what a higher-dose bound supports —
e.g., "a pre-registered replay control shows generic extra updates on live
groups buy a larger mean gain without the coverage cost (3/3 seeds), scoping
recycling's distinctive value to the all-fail regime." If E2 runs and B2 beats
matched replay, the stronger language comes back with evidence. Run E2 if the
shared A10G allows; do not let it gate the abstract.

**D3. Keep the title "The Estimator Decides."**
The review prefers the hedged title. But the project committed a title
decision rule (`01dd30c`) *before* wave 2: P-F2 confirmed → title stands. It
confirmed, 6/6 + 6/6. Overriding a committed decision rule on reviewer taste
reintroduces exactly the discretion the prereg discipline exists to remove.
The title names a registered, twice-replicated result. Keep it; make the
subtitle carry the scope ("...: Coverage Trade-offs in Curricula and Failure
Recycling for RLVR").

**D4. The GSM8K negative deserves one main-text paragraph, not only an
appendix.**
Agreed it exits the abstract, contributions, and main figures. But "we built
the steering-controlled experiment, committed telemetry gates, and the
treatment could not be delivered at this scale — twice, by 0.0015 on the
second try" is a *boundary result* that strengthens the paper's credibility
and warns off the obvious follow-up experiment. One paragraph in limitations
(the review's §9 limitations list already includes it — the review is
internally inconsistent in asking for appendix-only in §5).

**D5. Priority reorder under the real constraint (shared A10G, ~3 weeks of
runway).**
The review's GPU queue puts E2 first among new runs. Given D2, the corrected
queue is:

1. **E5 first if and only if the gate stays a contribution** — it is the only
   experiment whose outcome changes the paper's *claim structure* (third
   empirical contribution vs. appendix heuristic). Decide by 08-10: if the
   calibrated corrected-decay pilot can't be frozen by then, demote the gate
   now and skip E5 entirely. Demotion costs one contribution bullet; a failed
   late E5 costs the same bullet plus two weeks.
2. **E2 second** (upgrades wording strength; never changes paper identity).
3. E3 evaluation passes wherever checkpoints allow, CPU/analysis in parallel
   throughout (E1, E4, fig9 repair, verdict-artifact refresh are all
   GPU-free and unblocked today).

### II.3 Items the codex review missed

- **Premium reanalysis 24/24** — same correlated count, needs the E1
  treatment (see I.2.2).
- **Claims-and-evidence appendix table** (`body.tex:1727-1729`) repeats
  "24/24" and "10/12" — E1's language fix must propagate there and to the
  site (`docs/index.html`), which has previously drifted from the paper.
- **`e_llm1b_verdicts.json` embeds absolute machine paths** — a concrete
  instance of the review's own §11.9 sanitization item; there will be others.
  `grep -rn '/home/ec2-user' --include='*.json'` before the anonymized bundle.
- **The reconciliation debt is quotable by a reviewer**: `body.tex:1859-1861`
  admits the GSM8K pass@k harness sits ~3× below the trainer's val metric,
  unreconciled. With GSM8K demoted to an appendix this becomes tolerable, but
  the reconciliation script (ARM C) should still run to closure or the
  admission should say why it can't.

---

## Part III — My final review of the paper (referee form)

**Summary.** The paper derives an exact, estimator-specific "coefficient
mass" functional for success-conditioned RLVR estimators — for MaxRL's
deployed drop-all-fail estimator, per-task update mass is exactly
2(pass@N − pass@1), with a truncation-order correction (T = N−1) to the
original method's claim — and argues that this functional, not the curriculum
or the recycler, determines where sampled tasks emit learning signal. It
supports this with (i) exact frozen-schedule orderings with scheduler-mismatch,
no-SD, own-mass, and matched-lr controls; (ii) a preregistered, twice-run
balanced maze factorial whose registered primary (time-integrated coverage,
MaxRL > GRPO) confirmed 6/6 per sampler on fresh seed blocks after the
stronger endpoint form was retracted at its own registered test; and (iii) a
three-seed Countdown result where exact-verifier failure recycling raises
mean@16 while lowering pass@16 — with a replay control showing extra
live-group updates reproduce the mean gain without the coverage cost, and a
preregistered negative (Jugs) bounding the intervention family.

**Strengths.**
1. The theory-to-measurement chain is unusually tight for this literature:
   an exact algebraic object, tested at an exact rung with controls, then at
   a neural rung with preregistration and committed falsification branches
   that were *executed*, twice, against the authors' own headline claims.
2. The negative-result hygiene (retracted cohort claim, refuted gate dial,
   failed steering gates, Jugs null) is exemplary and, properly compressed,
   is a strength reviewers will reward rather than punish.
3. The practical takeaway — report pass@k beside mean accuracy, conditioned
   on the estimator — is actionable and well-earned by the Countdown result.
4. Artifact quality (one-command reproduce, frozen manifests, vendored
   integration code, run registry) is far above the venue norm.

**Weaknesses.**
1. *Independence unit.* Headline counts ("24/24 paired blocks," "10/12
   pairs") aggregate two correlated sampler contrasts per seed block. The
   block-level result (12/12, CI excluding zero) is strong and should be the
   quoted form; the easy-band localization does not survive block-level
   aggregation (CI includes zero) and must be downgraded to suggestive.
2. *Claim/label drift.* The abstract's "one validated operating point"
   (under-gated by a decay bug; corrected strong setting refuted) and the
   conclusion's "dose-matched replay" (2× dose) outrun the more careful body
   text. The strongest fix is already written in §6.8/§6.9 — propagate it up.
3. *Contamination disclosure.* SFT warmstart overlaps 21% of tier-0 eval;
   between-arm contrasts survive but the "zero train/eval overlap" sentence
   is incomplete as written.
4. *Single-seed figure.* The pass@k crossing figure transcribes one seed
   in-script while the reproducibility section claims no hard-coded figure
   values; the 3-seed endpoints exist, so this is repairable.
5. *Length and form.* ~17 pages of main text against a strict 9-page limit,
   2026 style file, no AI-use statement — the current document is a research
   report, not yet an ICLR submission.

**Verdict.** The underlying research clears the ICLR bar: an exact result, a
registered replication, an applied phenomenon with a cost metric the field
under-reports, and honest boundaries. The manuscript does not yet clear it.
Every blocking defect is repairable without new GPU results except optionally
the gate's status. As a referee I would currently rate this "major revision" —
with the E1/E3/E4 analysis fixes, the wording corrections, and the 9-page
restructure, it becomes a credible accept-range submission; with E5 or E2
landing positive it becomes a strong one.

---

## Part IV — Final guidance (the plan I would execute)

**This week (08-07 → 08-10), all CPU, all unblocked:**
1. E1: block-level factorial reanalysis → one JSON + one seed-block figure;
   propagate language to abstract, §6.3b, §6.4, conclusion, claims table,
   premium passage (n=12 form), site. Kill "24/24" and pair-level "10/12"
   everywhere; keep "6/6 per sampler, registered" and "12/12 blocks."
2. Wording fixes: abstract gate sentence → "one promising under-gated
   operating point; corrected-code replication open" (or delete if demoting);
   conclusion "dose-matched" → "higher-dose"; "not relabel-specific" →
   bounded form; SFT/tier-0 disclosure sentence + `data_integrity_check.json`
   update; delete the "none are hard-coded" clause.
3. Execute the g3p committed branch: §6.7 → limitations paragraph +
   appendix; refresh `e_llm1b_verdicts.json` (and strip absolute paths);
   update claims-table row to its falsification-branch text.
4. **Decide the gate's fate by 08-10** (D5). This is the only open decision
   that changes the paper's identity.
5. Move fig9 arrays to versioned JSON; start E3 derivation from the 3-seed
   raw outcomes; E4 clean tier-0 eval on the 101 tasks.

**08-10 → 08-28 (GPU, in this order):** E5 (only if gate retained, frozen
calibration first) → E2 (three seeds, matched accepted-group accounting) →
any E3 generation. Hard stop 08-28. Every run: prereg, registry row,
committed branches — the existing standard.

**08-29 → 09-17:** result lock; 9-page rebuild on `iclr2027_conference.sty`
with AI-use + reproducibility statements; the review §8 skeleton with title
kept per the committed rule (D3); red-team pass with the review's six
questions; anonymity sweep including JSON-embedded paths.

**09-18 / 09-25:** abstract, then full submission, each ≥12h before AOE,
verified by re-downloading from OpenReview.

**Bottom line.** I co-sign the codex review's diagnosis and most of its
prescription: the work is done; the accounting and the compression are not.
My amendments: report P-F3 as registered-bar-met with the unit caveat rather
than erasing it; treat E2 as strengthening rather than blocking; keep the
committed title; give the steering failure one honest main-text paragraph;
and decide the gate's status *now* rather than letting E5 dictate the
schedule. The paper this becomes — "the estimator decides what curricula and
recycling can do, and mean accuracy alone hides the coverage bill" — is
narrower than the project's ambitions and stronger than its current draft.

---

## Part V — GPU queue status and amended guidance (2026-08-07 ~21:15 UTC)

Checked live while pushing this document. This changes the near-term queue
recommendation in Part IV.

**Observed state:**

- `.done_h6_g3p` exists; the g3p repair driver exited cleanly at 18:01 UTC.
- A watcher armed on that marker (PID 2060603) then relaunched
  `smollm/run_h6_steered.sh`, whose two-pass OOM-retry loop is now **rerunning
  the E-LLM-1b cells that lack done-markers: g3s (training now, step 5/50 at
  ~27 min/step), then g3u, then m3s** — roughly 3 days of A10G time end to
  end.
- The OTG ablation driver (E-LLM-2c) launched at 18:13, hit the occupied GPU,
  died at vLLM engine-core init, and exited (its `flock -n` does not retry).
  The fullcv maze chain still waits behind OTG markers that will now never
  appear without a relaunch.

**The arithmetic that matters:** g3s's dead-sampled fraction averages 0.698
over its first 5 steps. To pass its preregistered delivery gate (run-mean
< 0.60) the remaining 45 steps would have to average below 0.589 — when g3p,
which adds the power-4 sharpener specifically to strengthen steering, only
achieved 0.601. **g3s is arithmetically near-certain to fail the same gate
it exists to pass.** g3u (repetition control) and m3s (MaxRL safety contrast)
are only interpretable *relative to a treatment-delivered cell*, and none
exists or is in prospect. The entire remaining steered queue is therefore
artifact-completeness only — it cannot change any paper claim, per the
committed P-S1 branch already executed in Part I.

**Amended recommendation (supersedes Part IV's queue for the next 72h):**

1. **Stop the steered pass-2 queue** (watcher PID 2060603 and the running
   `run_h6_steered.sh` tree) — or, at minimum, let the in-flight g3s cell
   finish and touch the remaining done-markers so g3u/m3s do not launch.
   I have NOT killed it: it belongs to a concurrent session's armed protocol,
   and stopping another session's preregistered run is an owner decision.
   But the numbers above say every additional GPU-hour it consumes is spent
   on a foregone conclusion, against a 2026-08-28 training hard stop.
   Record the archive decision in the run registry either way ("pass-2
   retry superseded by executed P-S1 inconclusive branch, 2026-08-07").
2. **The freed GPU goes to the Part IV order**: gate decision by 08-10 →
   E5 (only if the gate is retained) → E2 → E3 generation.
3. **Re-arm the OTG ablation driver** after the GPU clears (it preregistered
   P-OTG1/2 and crashed on contention, not on its own logic) and the fullcv
   chain behind it — both are committed preregs and cheap relative to E2/E5.
4. The GSM8K paper actions in Part I/IV are unchanged and unblocked: the
   verdict is already in hand; nothing running or queued can alter it.
