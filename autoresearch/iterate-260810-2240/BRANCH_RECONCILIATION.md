# Release-branch and E2c reconciliation handoff

## Why reconciliation is required

The active checkout is `main@81d4bd4` with a substantial, intentional dirty
worktree containing E1 repairs and the new E2c runtime. A later remote release,
`origin/codex/curriculum-maxrl-research@9277141`, contains the finished compact
paper and a much broader artifact release. Both carry valuable work; neither
should overwrite the other wholesale.

The release branch is the source of truth for the compact submission layout
(`paper/body_iclr.tex`, ICLR 2027 style/wrappers, release manifests, PDFs, and
large post-main experiment artifacts). The active worktree is the source of
truth for E2c (`curriculum_maxrl/countdown/`, the vendored replay changes,
runtime patches/launchers, and the frozen E2c preregistration).

## Reviewed ports already made

The following release-branch improvements were reviewed and selectively
ported/adapted into the active worktree:

- Countdown SFT/evaluation overlap audit, tests, and machine-readable summary;
- structured `fig9_bestk_proxy.json` and correct bootstrap-proxy terminology;
- manuscript/site demotion of the seed-1 curve and historical gate claims;
- reviewer-arm metric provenance; and
- awareness that the compact nine-page paper already exists.

The active worktree adds beyond the release branch:

- E1 independent seed-block reanalysis and plot;
- immutable-reservoir E2c training and delivery validation;
- exact replay-matcher optimization with equivalence coverage;
- raw-outcome endpoint recomputation and paired evaluation-provenance gates;
- GPU-safe orchestration for B1/B2 seed 3, reservoir, E2c seeds 1--3, delivery,
  then held-out evaluation; and
- the current goal, preregistration, status, and handoff documents.

## Files requiring deliberate conflict resolution

The principal overlaps are `README.md`, `EVIDENCE.md`, `SHARPENING_SYNTHESIS.md`,
`docs/index.html`, `paper/body.tex`, `paper/results/manifest.json`,
`reproduce.sh`, the Countdown overlap files, reviewer-arm provenance/extractor,
and the factorial provenance/verdict. The release versions should normally be
the textual base because they are more compact and cautious; then port the E1
block analysis, E2c protocol/status, raw-outcome metric clarification, and new
figure/audit manifest entries on top.

Do not copy generated PDFs or all release figures before selecting the final
source and Matplotlib/TeX environment. Do not treat Git conflict resolution as
scientific validation: rerun all checks after reconciliation.

## Recommended integration sequence

1. Preserve the current E2c worktree as an explicit patch or commit on a new
   integration branch; do not mix it into an anonymous frozen release commit.
2. Use `9277141` as the base of a fresh integration branch.
3. Apply only the active worktree's E1/E2c code, preregistration, tests, and
   current evidence corrections. Resolve the files listed above by meaning,
   not by choosing one side mechanically.
4. Keep `paper/body_iclr.tex` as the submission body and `paper/body.tex` as the
   extended record. Add the E2c verdict to the compact body only after all
   delivery gates and three endpoint evaluations complete.
5. Regenerate the final figures in the release's pinned environment, compile
   both wrappers with a TeX engine, and run the release verifier plus the E2c
   tests. Confirm the conclusion still ends on page 9.
6. Freeze a new content-addressed release only after manuscript numbers,
   structured artifacts, and raw-outcome recomputation agree.

## Non-negotiable scientific merge rules

- Historical Countdown `best@k` remains a bootstrap coverage proxy; never
  relabel it standard pass@k.
- E2c standard pass@16 comes only from retained binary outcomes after delivery
  passes for all three seeds.
- The historical B aggregate supplies no paired seed signs or timing claim.
- P-F2 uses seed blocks as the independent unit; sampler contrasts are repeated
  measurements within each block.
- The higher-dose two-epoch arm does not isolate relabel direction.
- The faulty-decay moderate gate is descriptive; corrected strong gating did
  not validate a useful operating point.
- E2/E2b remain treatment-delivery inconclusive and are never pooled with E2c.
