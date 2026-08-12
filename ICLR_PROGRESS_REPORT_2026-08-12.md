# ICLR track progress report — 2026-08-12

**Scope:** first execution pass of `CODEX_GOAL_ICLR_2026-08-12.md` (run directly by the issuing session).
**Bottom line:** Workstreams 2 and 3 are substantially complete — the draft compiles, fits the 9-page bound, has a 58/58 claim-to-artifact trace, and has absorbed a full adversarial referee pass. Workstream 1 (E2c) remains GPU-blocked with an automatic launch watcher now standing by.

## Workstream 1 — E2c

- Readiness receipt refreshed via `run_e2c_rtx5090.sh --readiness-only`: every integrity check passes; `launch_authorized_now: false` solely on GPU occupancy; next stage remains `train_e2_clean_b1_s3_260809`.
- Occupancy has been rising, not falling: 13,091 MiB at 00:30, 17,132 MiB at 00:40 (Cosmos critic daemon ~7.3 GiB plus holosoma/MuJoCo and other lab jobs). All external to this project; none touched.
- A watcher loop now polls the receipt every 10 minutes and will run the unchanged frozen driver the moment it self-authorizes (max 3 attempts, then stops for human review). Log: `$CLAUDE_JOB_DIR/tmp/e2c_watch.log`.
- **Human action required:** the RTX 5090 must drop below 4,096 MiB. Every day of continued occupancy consumes the runway to the Aug 28 training stop.

## Workstream 2 — Compile and page bound

- No TeX engine existed on this machine; installed tectonic 0.17.0 (static binary, no sudo) as a provisional engine. **Pinned-environment recompile at reconciliation is still required** before trusting the final page break.
- Vendored the two figures the compact body references that lived only on the release branch (`fig_countdown_core.*`, `fig_maze_block_contrasts.*`, additively via `git show`).
- Draft compiles clean: 0 errors, 0 undefined citations, ICLR 2027 style + times, AI-use statement present, inline bibliography intact.
- **The conclusion ends on page 9** (references and appendix follow; reproducibility and AI-use statements sit between conclusion and references). Achieving this took ~20 targeted compressions confined to the allowed cut order (related-work prose, evidence prose, conclusion redundancy) plus three modest figure-width reductions (.88→.84, 1.0→.94, .56→.52). No statistic, no limitation, and no terminology contract was touched.
- Compiled PDF (provisional): SHA-256 `0725f6c7280d75237826b11347423041b64d2037fa238d5ee9d5c150f6d3c5d8`.

## Workstream 3 — Verification and adversarial review

- **Claim trace (`paper/CLAIM_TRACE_ICLR.md`): 58/58 statistics traced** to checked-in structured artifacts (34 exact, 24 after rounding, 0 untraced). Seven artifacts that resolved only on the release branch were vendored into `main` (both Acrobot analysis JSONs + results doc, Digits confirmation analysis, N-sweep JSON + doc, maze leave-one-block-out JSON). One deliberate exception: the release's 562-row `run_registry.json` conflicts with a different local 53-row file of the same name — left for branch reconciliation, documented in the trace.
- **Forbidden-phrase scan:** empty, including the newly retired "recycling-package sharpening."
- **Terminology audit:** the historical Countdown scalar reads "VERL bootstrap best@16 (coverage proxy)" at every occurrence; standard pass@16 appears only in the E2c-protocol context.
- **LfH cross-check:** their relabeling triggers on a mean-reward threshold (signal repair), not an estimator-derived selection score; our related-work sentence claims nothing more. Closed.
- **Adversarial referee pass:** full report absorbed; predicted score 5 pre-fix with a named path to 6. All seven major findings fixed by edit (novelty self-contradiction on the "flat in N" claim, estimator-matching→rollout-awareness reframe, probe-cost transfer note, scale billing, coined-term retirement, SESOI reconciliation, sign-test granularity floor); minors fixed or logged. Full disposition: `REVIEW_RESPONSE_2026-08-12.md`.

## Workstream 4

- `paper/OPENREVIEW_ABSTRACT_CANDIDATE.md` created and synced to the post-referee abstract (title, plain-text abstract, keywords, TL;DR, conditional E2c update rule). Freeze deadline Sept 16.
- Corrected-code gate replication: still gated behind E2c per the goal document; drop decision due ~Aug 22 if not startable.

## Deviations from the goal document

1. The goal document reserved the compile for "the release's pinned environment." None exists on this machine, so a provisional tectonic compile was used to make page-fitting actionable now; the pinned-environment verification obligation is unchanged and restated here.
2. The trace-artifact vendoring slightly front-runs reconciliation, but only additively (no local file overwritten; the one conflict was left alone and documented).
