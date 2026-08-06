# Distance to an ICLR-ready submission (assessed 2026-08-03)

Reference bar: ICLR main text ≤ 9 pages (10 for camera-ready), unlimited
appendix; typical accepted empirical-RL papers carry 3+ seeds on headline
claims, complete related work, anonymized artifacts, and a reproducibility
statement. Assessment against the current 16-page draft (14 main + 2 back).

## Verdict in one line

The *science* is ~85% submission-ready (theory verified, headline result
pre-registered and multi-scale, negatives documented); the *manuscript* is
~60% ready — main text is 40% over budget, one LLM-scale replication is
still single-run-conditional, and formatting/anonymization work is undone.

## BLOCKING (cannot submit without)

- **B1. Page budget.** 14pp main → 9pp. The escalating-ladder section
  (6.1–6.9, ~7pp) is the target: each rung's Setup→Result→Takeaway can
  compress to ~½ column with full details in appendix. Plan: keep 6.1
  (ladder table + saturation), 6.3–6.4 (headline sign flip + mechanism),
  6.8–6.9 (sharpening + gate) at full length; compress 6.2/6.5/6.6/6.7
  to one paragraph each pointing to appendix sections. [~1 day of editing]
- **B2. ICLR class file + anonymization.** Currently article class with a
  GitHub URL on the title page and repo pointers throughout. Needs
  iclr2027 style, anonymized artifact link (anonymous.4open.science),
  and scrubbing "our repository" references. [half day]
- **B3. The LLM-scale interaction claim needs its decisive run.** The
  paper currently says "1-of-2 seeds, dose effect" — reviewers will
  probe exactly this. E-LLM-1b (g3s/g3u/m3s, treatment-delivered,
  pre-registered) is TRAINING NOW; its result lands either way as the
  §6.7 update (confirm → claim upgraded; refute → claim scoped to maze
  + the pool-conditionality story). [~2 days GPU, running]

## STRONG (reviewers will ding without)

- **S1. Second seed for the Countdown corrected-gate operating point**
  ~~(the full-strength point in Fig 8a is 1 seed)~~ **RESOLVED
  2026-08-06 by ARM A (3 seeds): the 1-seed point did NOT replicate —
  P-R1 refuted, Fig 7a redrawn as scatter.** Maze GRPO+teacher
  single-seed arm: superseded by the balanced factorial (6 blocks both
  samplers ×2 waves).
- **S2. Reference hygiene**: 4 entries still cite arXiv IDs with "et al."
  reconstructed from abstracts — verify against the PDFs; LILO/SFL/DUMP
  entries lack arXiv IDs; dapo bibitem says NeurIPS 2025 (check venue).
- **S3. Figure 4/5 (algorithm + partition map) refer to §ordering that
  changed; re-audit all \S references after B1 restructure.**
- **S4. RESOLVED 2026-08-06:** Jugs entropy-collapse figure added to the
  appendix (fig10_jugs_entropy, all 9 trajectories, manifest entry) with
  the rollout-set-diversity design gate stated in the caption + limitations.

## NICE (improves odds, not required)

- N1. A pass@k-vs-k sweep figure for the maze (the crossing-at-k≈4
  result is currently prose).
- N2. Countdown gate_max_p dose sweep — **CLOSED 2026-08-06: the
  fixed-decay designed-strength sweep (ARM A) refuted the dose-response
  reading (P-R1); there is no curve to draw. Fig 7a is a scatter.**
- N3. Whittle-index theory paragraph (parked; only if a theory reviewer
  is anticipated).

## What is already at or above the bar

- Prop 1 + Lemma 1 machine-verified; MC scripts committed.
- Headline safety result: 9 runs, exact permutation p=0.0079, direction
  replicated 2/2 seeds at LLM scale.
- Pre-registration discipline documented with timestamps and one spent
  prereg (E-LLM-3) correctly recorded as nulls + postmortem.
- Negative-results section with mechanisms, not apologies.
- Appendix A hyperparameter/knob tables; compute statement; per-figure
  regeneration from committed JSON.

## Order of work (while E-LLM-1b trains)

1. B1 restructure (biggest lever, no GPU dependency) — start now.
2. B2 class file + anonymization pass immediately after.
3. S2/S3 hygiene sweeps on the restructured text.
4. When E-LLM-1b lands: fold the P-S1..P-S3 verdicts into §6.7 (either
   direction), then S1 seeds, then freeze for internal review round 4.

## Progress log

- 2026-08-03 pass 1: App B created (full details for compressed rungs);
  §6.1/6.5/6.6/6.7/6.9 compressed; fig6_gym → appendix. Main text
  14pp → 13pp. Remaining ~3pp must come from: §3 interpretations
  (tighten), §6.3 (split attribution detail to App B), Q2/Q3 intro
  compression, and the ICLR two-column-free format change itself
  (article→iclr style typically saves ~10% through tighter spacing).
  Note: B2 blocked locally — no ICLR .sty on this machine; vendor
  iclr2027_conference.sty into paper/ when network fetch is possible,
  or hand off to the user.
- Sentence pass: worst dash-chains split (Q1), "honestly read" label
  removed; §6.7/6.9/related-work chains remain (4-9 dashes/paragraph,
  mostly structural lists — acceptable) — revisit after E-LLM-1b text
  lands.
- 2026-08-03 pass 2: rem:scope, §6.3 efficiency detail, Q2 compressed;
  hardcoded §refs → \ref. Main text 13pp → 12pp.
- 2026-08-03 B2 DONE: main_iclr.tex builds with vendored
  iclr2026_conference.sty — double-blind header, line numbers,
  author-year citations, bib-before-appendix ordering, no repo URLs.
  12pp main in ICLR format (target 9-10; remaining compression:
  §3 interp blocks, §4/§5 prose, fig1 sizing).
- E-LLM-1b status: g3s OOM'd at step 3 (node RAM); queue hardened
  (ray cleanup between cells, done-markers on global_step:50, retry
  pass); amendment A1 adds g3p (warm-start + power=4) after the g3s
  partial showed warm-start-alone doesn't move dead-sampled off the
  population rate. m3s training now; g3s/g3p/g3u follow.

- 2026-08-04 pass 3: §4 opening, §5 interp+contracts, one-identity
  corollary tightened; fig1 0.92x, fig5 0.5x; all compressions synced
  into both editions (main.tex working draft + main_iclr.tex). ICLR
  main text holding at 12pp — the last ~2pp are float-spacing bound
  (pages 5-6 carry two half-empty float pages), not prose: next lever
  is float packing ([t] -> [h]/[tb] consolidation + wrapping Alg 1 and
  fig2 onto one page), then §6.7's remaining length after E-LLM-1b
  verdicts replace the interim seed-2 text.

- 2026-08-04 S2 CLOSED: all 15 bibliography entries web-verified;
  5 corrected (GRESO title+venue, DisCO title, SC-SDPO author, 
  Reinforce-Ada retitle, LILO author list). N1 CLOSED: fig9_ksweep
  (maze crossing) added to App B. Site + both PDFs pushed for review.

- 2026-08-04 expert-guidance pass (PR #2 merged; 54pp review under
  research_guidance/2026-08-04/). Prose P0s resolved in the parallel
  math-review pass (69b7704). New EXPERIMENTS closing the guidance's
  evidence gaps, all committed with artifacts:
  - verify_guidance_math.py — MC-confirms the three-estimator table
    (raw/full-CV/practical masses), T=N-1 gradient identity, the
    factorization E[g]=nu_N(p)(mu+−mu−), the nonzero full-CV all-fail
    update ∇p/q, the exact peak, and the DEPLOYED sample-SD GRPO tail
    ratios (√N hard, (N−1)/√N easy) that replace the population-SD
    √(N−1) claims.
  - run_fullcv_baseline.py (P0.5/P1.1) — the full control-variate
    all-fail baseline the guidance demanded before saying "only
    channel": full-CV scores 0.000 in every seed on the frontier-heavy
    pool while recycling ignites either variant to 0.98; §6.2 now
    reports it.
  - run_schedule_matched.py (P1.2, exact rung) — frozen realized
    schedules replayed under MaxRL/GRPO/RLOO from identical inits:
    estimator coverage ordering survives schedule matching 10/10
    paired contrasts; §6.3 now points to it as the schedule-matched
    core of the confirmatory factorial.
  - run_proposal_shift.py — hindsight proposal-law diagnostics
    (cos 0.93 vs exact fresh-destination direction, p_Q−p_Π=+.003)
    + cross-fitted destination selection (alignment 0.82, AUC
    .883→.829: adaptive reuse is mildly helpful here); cited in the
    hindsight remark's interpretation.
  - run_transfer_matrix.py — exact activity-vs-transfer matrix;
    Spearman(activity, one-step pool value)=1.0 on chains at 3
    snapshots; rem:scope now scopes the tie to shared-prefix pools.
  Remaining from guidance: GPU items unchanged (E-LLM-1b running,
  corrected-gate 3-seed sweep, GRPO-own-teacher + no-std arms queued);
  balanced maze factorial (≥6 seed blocks) still the big open design.

- 2026-08-05: Review-round-4 resolution complete. All text-fixable items
  from 5 reviews FIXED (see reviews_round4/RESOLUTION_MATRIX.md); part J
  (flat-over-band control) run and folded into the Remark — the zeros
  alone forfeit the gain (0/10), settling R5-Q3 with data; deployment-
  limits paragraph (R2-W3/W4) + wiring contract (R2-Q4) shipped; site
  aligned with rescoped claims. Runability audit passed on all queued
  arms (syntax, estimator edge cases, g3p power path, data deps, disk).
  REMAINING (all experiment-gated): E-LLM-1b verdicts -> 6.7 rewrite;
  GRPO-own/no-std maze arms -> title-claim scope; ARM A/B/C ->
  gate-dial + dose-baseline + harness paragraphs; then the section-6
  restructure + claims table, final float pass, camera-ready hashes.

- 2026-08-06: ARM A COMPLETE, **P-R1 REFUTED** at 3 seeds (mean-kept
  −0.26, window [0,.60]; coverage .525 vs floor .541): falsification
  branch executed — Fig 7a redrawn as operating-point scatter, §6.9 +
  fig9 caption + conclusion + site de-dialed; artifacts vendored to
  curriculum_maxrl/countdown_reviewer_arms/ (commit 0fd5f70). ARM B
  interim (2/3 seeds): replay ppo_epochs=2 EXCEEDS recycling on both
  axes (t1 mean .467 vs .324; pass16 .629 vs .492) — §6.8 carries the
  dose-control paragraph with the dose caveat; final verdict on s3.
  REMAINING (experiment-gated): ARM B s3 -> P-R2 final; g3p ->
  E-LLM-1b verdict -> 6.7 rewrite; maze grpo_mass/nostd/fullcv sweeps
  (queued); ARM C harness reconciliation; OTG ablation (P-OTG1/2).
  Then: section-6 restructure + claims-vs-evidence table, page
  compression toward 9pp, camera-ready hashes, anonymized artifact
  mirror.
