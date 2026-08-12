# Goal statement for codex — ICLR 2027 track, completion phase

**Date issued:** 2026-08-12
**Companion document:** `CODEX_GOAL_ICRA_2026-08-11.md` (ICRA track; independent, CPU-only, do not mix)
**Governing documents (frozen, read before acting):**
- `autoresearch/iterate-260810-2240/STATUS.md` — live E2c state and stage order
- `autoresearch/iterate-260810-2240/E2C_PREREG.md` — frozen protocol; immutable
- `autoresearch/iterate-260810-2240/BRANCH_RECONCILIATION.md` — merge rules for the release branch
- `FINAL_ICLR_REVIEW_AND_COMPLETION_GUIDE_2026-08-07.md` — claim ledger and statistical rules
- `RESEARCH_REVIEW_HANDOFF_2026-08-11.md` — non-negotiable terminology
- `LITERATURE_POSITIONING.md` — novelty defense and related-work content

**Deadline chain:** E2c training hard stop **Aug 28** → ICLR abstract **Sept 18** → full paper **Sept 25** (9-page strict initial-submission limit).

## Current position (verified 2026-08-12)

- The submission body is `paper/body_iclr.tex` (extracted from release `origin/codex/curriculum-maxrl-research@9277141`, polished and committed at `eaa05fa`): calibrated abstract, corrected claim hierarchy, concurrent-2026 related-work cluster with six verified citations.
- E1 (block-level factorial reanalysis) is **done**. E3/E4 (artifact repairs) are **closed** at the limit of surviving historical data.
- **E2c is the only remaining experiment** and the critical path. B1/B2 seeds 1–2 are complete and audit-passed for reuse. Remaining, in frozen order: B1+B2 seed 3 → frozen-SFT reservoir generation + static preflight → E2c replay seeds 1–3 → three-seed delivery validation → paired step-60 nine-arm held-out endpoint.
- E2c launch is **blocked solely on GPU occupancy**: the driver refuses to start above 4,096 MiB on the shared RTX 5090, and unrelated processes (Cosmos critic daemon ~7.3 GiB; previously OpenPI eval ~4.5 GiB) hold it.
- The nine-page bound of the polished draft is **unverified** — no TeX engine exists on this machine; compilation must happen in the release branch's pinned environment.
- LfH (arXiv:2607.09042) full-text read is done (`LFH_NOTES.md`, 2026-07-28).

## The goal of this phase

**Deliver a compiled, page-compliant, adversarially reviewed ICLR submission whose every number traces to a structured artifact — with the E2c endpoint included if the GPU frees before Aug 28, and with the preregistered-but-blocked protocol honestly disclosed if it does not.** Four workstreams, in priority order.

## Workstream 1 — E2c execution (critical path; highest priority)

1. Refresh the outcome-blind receipt regularly:
   `bash verl_integration/run_e2c_rtx5090.sh --readiness-only`.
2. When the receipt reports `launch_authorized_now: true`, run the **unchanged** driver: `bash verl_integration/run_e2c_rtx5090.sh`. Its stop rules enforce the stage order; do not modify seeds, steps, ceiling, paths, gates, or the MaxRL commit — the preflight rejects drift by design.
3. **Never kill, pause, or nice the occupying processes.** The Cosmos daemon and OpenPI jobs are outside this project's scope. If the GPU is still occupied, report the occupancy (process, MiB, receipt SHA-256) to the user and stop — freeing the GPU is a human decision.
4. No substitute is acceptable: no smaller model, CPU surrogate, relaxed ceiling, partial endpoint, or replacement protocol. The frozen goal completes only via the unchanged driver.
5. **Aug 28 decision rule:** if all training stages (through E2c replay seed 3) are not complete by end of Aug 28, stop launching. Write a dated closure note recording exactly which stages completed, and the paper ships on the historical-proxy-plus-disclosed-protocol basis already written into `body_iclr.tex` ("until that protocol completes, we report the logged proxy under its actual provenance"). Endpoint evaluation of already-trained stages may continue past Aug 28 (it is evaluation, not training).
6. If the endpoint completes: run the sealed nine-arm analysis, then update the paper's Countdown section per the prereg's reporting plan — seed-level paired contrasts, mean@16 and standard observed-set pass@16 from retained binary outcomes, all three seeds shown. Whatever the direction of the result, it goes in; the prereg decides the wording, not the outcome.

## Workstream 2 — Reconciliation and compile

1. Follow `BRANCH_RECONCILIATION.md` exactly: file-level sequence, never a blind merge, git resolution is not scientific validation. The release branch is source of truth for the compact submission scaffolding; this worktree is source of truth for E2c and the E1 reanalysis.
2. Build `paper/main_iclr2027.tex` (wraps `body_iclr.tex`, ICLR 2027 style) in the release's pinned TeX environment. **Verify: main text ends on page 9 or earlier**, references excluded per ICLR rules; figures render; no missing citations (six new bibitems: groupstd, actorcurator, mopps, lze, passkinv, hir).
3. Confirm the ICLR 2027 style file is actually used (not the 2026 style) and add the required **AI-use / LLM-usage statement** per the ICLR 2027 call.
4. If over 9 pages, cut in this order: compress the Related Work paragraphs (keep the empty-cell opener and the concurrent-analyses paragraph), tighten Evidence prose around tables, move secondary maze detail to appendix. Never cut: the claim-hierarchy statement, the block-level statistics, the proxy-provenance disclosure, the limitations.
5. Rebuild `paper-iclr.pdf` and update the website link (`docs/index.html`) only after the compiled draft is final — the currently linked PDF is stale and that is known.

## Workstream 3 — Verification and adversarial review

1. **Number-to-artifact trace:** every statistic in `body_iclr.tex` must be generated from a checked-in structured file (`block_reanalysis.json`, `paper/results/manifest.json`, fig data JSONs, E2c endpoint artifacts if present). Produce a short trace table (claim → artifact path → value) as `paper/CLAIM_TRACE_ICLR.md`.
2. **Forbidden-phrase scan** (must return empty): "dose-matched", "pass@16" applied to the historical Countdown aggregate (correct term: "VERL bootstrap best@16 coverage proxy"), "24/24", "10/12 confirmed", "validated operating point", "expected learning signal", any "ALP-GMM" label on the learnability baseline.
3. **Statistical-unit audit:** independent unit is the training seed / seed block everywhere; sampler contrasts are repeated measurements; no pooled confirmatory p across waves; easy-band localization stays descriptive.
4. Run a full adversarial referee pass on the compiled PDF (strongest-critic mode: novelty vs the six concurrent works, the proxy disclosure, the single-LLM-domain scope, the N-ablation coverage). Fix what is fixable; log the rest as known risks in a dated review-response file.
5. Cross-check one positioning point from `LFH_NOTES.md`: confirm LfH contains no estimator-derived *selection* rule (their VLM relabeling is signal *repair*, not task pricing), and that our related-work sentence reflects that accurately.

## Workstream 4 — Optional strengthening (only if Workstreams 1–3 are green)

- **Corrected-code gate replication (P1):** the useful saturation-gate operating point ran with buggy decay; the review allows it only as "provisional mitigation." A three-seed replication at the moderate point with fixed code would upgrade it. Requires GPU — therefore strictly AFTER E2c completes, never before, and only if calendar allows (needs to finish training by Aug 28 too; if not startable by ~Aug 22, drop it and keep the provisional wording).
- **Abstract candidate (due Sept 18):** prepare the OpenReview abstract + title from the current calibrated abstract; freeze by Sept 16.
- Do NOT add new domains, larger models, or new experiment families. The review is explicit: the remaining work is accounting, repair, and compression.

## Hard constraints (all workstreams)

1. The RTX 5090 is E2c's; the ICRA campaign and any other job never touch it.
2. Never modify frozen preregistrations except via dated amendment sections.
3. Never relabel the historical proxy as standard pass@k; E2c pass@16 only from retained binary outcomes after all delivery gates.
4. ARM B is "higher-dose replay control," never "dose-matched."
5. No branch switching in this checkout; extract release files via `git show branch:path` additively.
6. Never commit `.codex/`. Commit at workstream boundaries with clear messages.
7. Report every milestone in a dated progress file (pattern: `ICLR_PROGRESS_REPORT_<date>.md`): what ran, artifact paths + SHA-256, deviations with justification.

## Acceptance criteria for this phase

- [ ] E2c either complete through the nine-arm endpoint, or closed by the Aug 28 rule with a dated closure note — no third state.
- [ ] Compiled PDF in the pinned environment; main text ≤ 9 pages; ICLR 2027 style; AI-use statement present.
- [ ] `paper/CLAIM_TRACE_ICLR.md` exists and covers every statistic in the body.
- [ ] Forbidden-phrase scan empty; statistical-unit audit clean.
- [ ] Adversarial review pass done, response file committed.
- [ ] Website PDF link current with the compiled draft.
- [ ] Abstract + title frozen by Sept 16 for the Sept 18 deadline.
