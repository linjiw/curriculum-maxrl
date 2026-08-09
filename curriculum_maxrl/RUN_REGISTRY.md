# Run registry

`run_registry.json` is the generated, one-row-per-training-run inventory for
the paper's maze, Countdown, GSM8K, and Acrobot evidence. It replaces the earlier
53-row mixture of runs and aggregate artifacts.

Regenerate it with:

```bash
python3 curriculum_maxrl/build_run_registry.py
```

Validate that the committed file is current with:

```bash
python3 curriculum_maxrl/build_run_registry.py --check
```

The check fails on duplicate run IDs, stale counts, malformed JSONL evidence,
missing repository-relative evidence/raw paths, incomplete Countdown reviewer
arms, unexpected factorial cell counts, malformed Acrobot source-lock/gate/raw
manifest bindings or run accounting, stale content hashes, or machine-local
absolute paths.

## Accounting

The 562 rows comprise:

- 94 maze runs: 31 historical matched-clock cohort runs, 3 checkpoint
  extensions, 36 wave-1 factorial cells, and 24 wave-2 factorial cells;
- 20 Countdown runs: 4 shallow-pool predecessor cells, 9 main sharpening
  runs, 1 corrected-gate follow-up, and all 6 reviewer-control runs;
- 7 GSM8K runs: 6 original-factorial/replication cells and the completed
  steering-controlled `g3p` cell; and
- 441 Acrobot runs: 40 historical V3 runs (20 paired seeds each under uniform
  and exact `u_16` sampling), 9 source-locked V2 tournament development runs
  (3 arms × 3 logical seeds), and 60 source-locked V2 confirmatory runs (3
  arms × 20 logical seeds), plus 12 ProCuRL-selection development runs (4 arms
  × 3 logical seeds) and 320 ProCuRL-selection confirmatory runs (4 arms × 80
  logical seeds). Quick smokes and invalid, aborted artifacts are excluded. V3
  remains historical/descriptive because a later audit found neighboring
  cross-domain RNG-root reuse; both fresh studies use the globally separated
  logical seeds recorded in their sealed protocols.

By raw-evidence availability, the registry contains 121 vendored aggregate run
records, 320 external content-addressed aggregate run records, 33 individually
vendored artifacts, and 88 other summary-backed or external runs. Thus all 441
Acrobot rows have aggregate-run-record accounting: the first 121 are vendored,
while the final 320 bind to an ignored external raw aggregate through per-run
canonical hashes.

The registry includes a run when a versioned local source establishes that the
run completed and is used by the paper.  `evidence_path` always exists in this
repository.  `evidence_locator` identifies the run within an aggregate source
when an individual artifact is unavailable. The legacy field name `raw_path`
means a vendored run-level file or aggregate containing a complete run record;
some files are extracted endpoint rows, not original trainer logs.
`raw_status`, `raw_locator`, and the file's schema disclose which. In
particular, the first 121 Acrobot rows point into four vendored aggregates that
retain each run's raw group diagnostics and evaluation curves. The 320
ProCuRL-selection confirmation rows point to vendored per-seed diagnostics and
to a content-addressed manifest for the external 1.37 GB raw aggregate; each
row carries its own canonical raw-run digest and byte count. The external
artifact's download URI is `null`. Confirmatory rows also point to their
derived analyses and portable verification receipts.
Some summary-backed rows also carry a `result_path`/`result_locator` pair for
the aggregate endpoint table.  An unknown seed is `null`, never inferred.
Affected Countdown rows also carry `metric_provenance_path`, which defines the
legacy `pass16` fields as VERL bootstrap best@16 proxies while leaving the
historical run-level files byte-for-byte unchanged.

Protocol labels such as “preregistered” are transcribed from the source
evidence records. The generator verifies the canonical sealed source-lock and
development-gate bindings for both fresh Acrobot studies, and additionally
checks the ProCuRL-selection raw manifest, compact diagnostics, portable
receipt, exact audited null result, and every arm/seed index. Because several
other locking objects are absent, the registry does not independently establish
their timing.

`n_eval_rows`, when present, is the count of serialized evaluation records in
the vendored run record.  It is not the number of prompts or generations.

## External artifact gaps

The top-level `external_artifact_gaps` section records every counted run whose
underlying execution artifact is not local.  In particular:

- all 60 balanced-factorial raw logs remain in execution fork `9f7dd2e`, while
  per-cell summaries and protocol documents are vendored;
- eight legacy maze raw logs are represented by vendored per-seed summaries;
- the primary Countdown arms have aggregate endpoints locally, while all six
  new reviewer-control run-level endpoint files—including replay seed 3—are
  vendored; these are not full trainer logs;
- the completed GSM8K `g3p` result is locally evidenced, but its raw log and
  refreshed `e_llm1b_verdicts.json` are not vendored; and
- the 320-run ProCuRL-selection confirmation raw is retained externally with
  exact size/SHA-256 and one canonical digest per run, while its independent
  analysis and descriptive diagnostics are vendored; no download URI is yet
  available; and
- no completion artifact is local for the superseded `g3s` pass-2 retry or its
  queued controls, so those jobs are not counted as completed runs.

The legacy registry at commit `0fd5f70` also named
`ck_falp_hsd_s1.jsonl`.  No raw artifact, versioned endpoint summary, or
current paper use is present locally, so schema v2 records it under
`not_counted_runs` rather than asserting that it is a completed run.

No EC2, macOS, Ray-session, or temporary absolute path is copied into the
registry.
