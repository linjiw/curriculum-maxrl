# ICLR 2027 pre-submission checklist — 2026-08-26

This is an operational checklist, not scientific evidence. Dates are computed
in America/New_York on 2026-08-26: 23 days remain to the 2026-09-18 AOE
abstract deadline and 30 days to the 2026-09-25 AOE paper/supplement deadline.

## Green now

- [x] **Canonical wrapper/style:** `paper/main_iclr2027.tex` uses the local
  `iclr2027_conference` style with final-copy mode disabled and anonymous
  authors. Both compact wrappers carry the PI-selected count-law title.
- [x] **Main-text edge:** compact PDF has 20 pages total; the conclusion ends
  on page 9 and references begin on page 10. The reproducibility and AI-use
  statements follow the conclusion.
- [x] **Build diagnostics:** the canonical compact log contains no undefined
  references, undefined citations, overfull boxes, emergency stop, or fatal
  error. Remaining diagnostics are underfull boxes and the vendored
  `algorithm*.sty` UTF-8 warning.
- [x] **Exact reproduction:** strict pinned mode passes all tests, all 17
  manifest inputs, all eight byte-exact compact figures, stored verdict spot
  checks, and both byte-exact manuscripts.
  - compact/web SHA-256:
    `8e337e232762aa425129371926d7da7d35414b17e37c20ae69070a62957da0d6`
  - extended SHA-256:
    `dcb2b6f16969a465e5c8259f605a9617283a436faf063a6dd444c12ecd467907`
- [x] **Mandatory scientific tests:** mass formulas, note claims, Countdown
  overlap, VERL curriculum, frontier framework, group-law, count-law stats,
  relabel-degeneracy, and E4 analyzer tests pass. Focused pytest totals were
  116 passed/10 skipped and 98 passed/1 skipped.
- [x] **Claim trace:** one unambiguous total, 97/97 traced and zero untraced.
- [x] **Clean-clone build:** a `git clone --no-local` of commit
  `8cef1a388004` passed portable `reproduce.sh --build` and regenerated the
  exact compact/web and extended hashes above.
- [x] **Reviewer pass:** five paper-blind-precommitted simulated reports and a
  mechanical synthesis are filed under `reviews_round5/`. No mandatory
  `block` was assigned. All adopted wording/scope repairs and rejected
  experiment requests are recorded in `reviews_round5/07_TRIAGE.md`.
- [x] **Cross-surface agreement:** paper, OpenReview candidate, README, and
  website use the same title, P0 verdict, aggregation-law assumption, and
  evidence boundary.
- [x] **Anonymous export:** two independent history-free allowlisted exports
  from commit `8cef1a388004` produced the same 4,263,540-byte archive:
  `75c2efc8cbcf6d36b52a2b91559fdd78abf8a8fdd673b1938b8d38690ed1824c`.
  Each contains 384 files/22 PDFs, passes portable reproduction, and passes
  host-path, identity-URL, PDF metadata/text, and symlink scans.
- [x] **PDF metadata:** compact PDF has no Author, Title, Subject, or Keywords
  value; creator/producer fields are generic LaTeX/xdvipdfmx identifiers.

## Open but bounded

- [ ] **E4 freeze or drop — due 2026-09-03:**
  `llm_calibration/LLM_COUNTLAW_CALIBRATION_PREREG.md` is a draft only. PI must
  approve the checkpoint, buckets, sample size, settings, statistic, and
  reporting perimeter; a fail-closed launcher/environment lock and final hash
  rebinding are still required. No sampling is authorized before the clean
  freeze commit. Hard stop remains 2026-09-08.
- [ ] **Anonymous deposit/DOI — PI-owned:** rerun the exporter at artifact
  freeze, upload only the history-free archive through an anonymous venue, and
  bind the checksum-matched DOI. Current DOI is null and no placeholder is
  cited.
- [ ] **External raw-data release decision — PI-owned:** P0, Acrobot, maze,
  paid-probe, Digits, and Countdown boundaries are now explicit. Decide which
  checksum-bound raw payloads can be deposited anonymously; do not describe
  unavailable raw reanalysis as one-command complete.

## Human/account checks — cannot be verified from this repository

- [ ] **Authors and order:** PI confirms the complete author list and ordering.
- [ ] **OpenReview profiles:** every author confirms a complete, current,
  conflict-correct profile before the 2026-09-16 lock.
- [ ] **Reciprocal-review eligibility:** PI verifies each author's ICLR 2027
  reciprocal-review obligations and eligibility in OpenReview. Repository
  state cannot establish this; it must not be marked green from a local scan.
- [ ] **Final form copy:** PI pastes the locked title/abstract from
  `paper/OPENREVIEW_ABSTRACT_CANDIDATE.md` and independently checks the rendered
  OpenReview fields before submission.

## Freeze sequence

1. By 2026-09-03, freeze E4 prospectively or record that it is dropped.
2. By 2026-09-08, stop E4 and integrate at most its preregistered Tier 2-prime
   appendix/table sentence if complete; otherwise record no evidence.
3. By 2026-09-12, freeze the 97-row claim table and evidence perimeter.
4. By 2026-09-16, lock title, abstract, authors, order, profiles, and reciprocal
   reviewing eligibility.
5. By 2026-09-22, freeze the full draft; afterward allow correctness,
   anonymity, references, figures, and artifact checks only.
6. By 2026-09-24, rerun exact reproduction, anonymous export, identity scans,
   and clean-clone build from the final release commit; disclose any remaining
   external dependency exactly.
