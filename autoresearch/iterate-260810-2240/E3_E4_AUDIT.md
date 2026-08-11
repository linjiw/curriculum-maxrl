# E3/E4 artifact availability and decision audit

**Audit date:** 2026-08-10 23:25 America/New_York  
**Outcome access:** no E2c held-out outcome was generated or inspected.

## Repository-state finding

The checked-out branch is `main` at `81d4bd4`, but the remote branch
`origin/codex/curriculum-maxrl-research` contains a later release commit,
`9277141` (2026-08-09). That commit already contains a nine-page ICLR
candidate, an artifact-recovery audit, a Countdown overlap tool, and the
correct classification of the historical Countdown coverage scalar as VERL's
with-replacement bootstrap `best@k` proxy. The branch is much larger than the
active E2c worktree and was not merged automatically. Only the narrowly
relevant audit/tooling changes were ported with review.

## E3: multi-seed pass@k curve

### Available

- `b_scoreboard_3seed.json` contains tier-level three-seed means and dispersions
  for `mean@16` and the historical VERL `best@16` scalar.
- Seed 1 has five transcribed `best@k` points for each displayed arm at
  `k={1,2,4,8,16}`.
- The transcribed points and their session labels now live in the declared,
  checksummed input `paper/figures/data/fig9_bestk_proxy.json`; the figure script
  no longer hard-codes them.

### Missing

- B1/B2/B3 seed 2--3 per-task outcomes and full per-k telemetry.
- All historical B1/B2/B3 task identities and 16 binary verifier outcomes.
- The referenced July Ray sessions and compatible historical checkpoints in
  the current filesystem.

Searches covered current and all reachable Git objects, the runtime checkpoint
tree, `/tmp/ray` session names, and plausible JSON/JSONL/log files under the
workspace and `/data/robotixx`. The later release branch independently records
an exhaustive recovery audit with the same negative conclusion. Aggregate
means and bootstrap summaries are not invertible to per-task outcome vectors.

### Decision

The seed-1 curve remains descriptive only. It is labeled a VERL bootstrap
`best@k` proxy, not standard unbiased pass@k, and the manuscript no longer
uses the crossing, paired timing, or per-seed signs as inferential evidence.
E3 cannot be completed from surviving historical artifacts. A replacement
requires new checkpoint inference or a new run that retains raw outcomes.

## E4: clean historical tier-0 evaluation

### Available

The original source audit reported, under identity `(target, sorted operands)`,
27/128 tier-0 evaluation tasks exposed during SFT, versus 0/128 in tiers 1 and
2. `curriculum_maxrl/data_integrity_check.json` now records those counts,
their non-vendored provenance, and the exact blocker. The restored
`audit_countdown_sft_overlap.py` can recompute overlap and clean endpoints from
JSON/JSONL/Parquet source exports, and its four tests pass.

### Missing

The historical SFT manifest, frozen evaluation manifest, per-task outcomes,
and compatible B-arm checkpoints are unavailable in this checkout. The clean
101-task tier-0 result therefore cannot be computed from the aggregate table.

### Decision

Historical full-128 tier-0 absolute values remain contaminated and excluded.
The absence of overlap in tiers 1--2 limits this particular leakage issue but
does not repair the missing raw-outcome/proxy problem. The rebuilt E2/E2c split
and clean-SFT checkpoint have zero SFT/test overlap in every tier; their new
evaluation retains raw outcomes, but those results constitute a prospective
replacement rather than a retrospective repair of the historical B study.

## Research consequence

E2c is more valuable after this audit, not less: in one prospectively gated
experiment it can test the relabel-direction question on the clean split and
replace the historical bootstrap proxy with standard raw-outcome pass@16. Its
analyzer now recomputes `mean@16` and pass@16 from all retained binary outcomes,
checks the held-out data hash and decoding settings, and enforces within-seed
paired evaluation provenance before reporting any contrast.
