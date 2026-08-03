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
  (the full-strength point in Fig 8a is 1 seed) and for the maze
  GRPO+teacher arm (single-seed, labeled). GPU-cheap (~8h total), queue
  after E-LLM-1b.
- **S2. Reference hygiene**: 4 entries still cite arXiv IDs with "et al."
  reconstructed from abstracts — verify against the PDFs; LILO/SFL/DUMP
  entries lack arXiv IDs; dapo bibitem says NeurIPS 2025 (check venue).
- **S3. Figure 4/5 (algorithm + partition map) refer to §ordering that
  changed; re-audit all \S references after B1 restructure.**
- **S4. The Jugs pool-conditionality paragraph cites repo postmortem —
  after B1, promote its mechanism (rollout-set diversity as a design
  gate) into the appendix with the entropy trajectories figure.**

## NICE (improves odds, not required)

- N1. A pass@k-vs-k sweep figure for the maze (the crossing-at-k≈4
  result is currently prose).
- N2. Countdown gate_max_p dose sweep (pre-registered as standing
  follow-up; would turn Fig 8a's 3-point frontier into a curve).
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
