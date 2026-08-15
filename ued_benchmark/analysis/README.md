# Fail-closed matched-development analysis

`development_protocol_v1.json` and `../UED_MATCHED_DEV_PREREG.md` freeze the
first 4x8/buffer-500 Frontier-versus-MaxMC development comparison.
`preregistered_dev_analysis.py` verifies a sealed campaign and all ten atomic
schema-2 run packages before parsing any numeric aggregate CSV value. Each run
package retains and revalidates the training-sidecar and evaluation-package
`SHA256SUMS`/`COMPLETE` closures, both receipts, the run context, all 30 ordered
raw episode records, and the original checkpoint/log/resource artifacts. It
intentionally cannot analyze partial campaigns, periodic checkpoints, resumed
runs, mismatched transition budgets, unbound files, or drifted
source/config/environment identities.

`../scripts/assemble_matched_run.py` is the only supported bridge from the
closed trainer/evaluator outputs into that schema. The campaign provenance
must freeze its SHA-256 under `assembler_driver_sha256`. It validates in a
temporary sibling directory and publishes with one atomic rename only after
the analyzer contract and raw-to-aggregate reduction pass. Its required CLI
is:

```bash
python3 ued_benchmark/scripts/assemble_matched_run.py \
  --campaign-manifest /absolute/path/campaign-manifest.json \
  --expected-campaign-sha256 <sha256> \
  --run-context /absolute/path/run-context.json \
  --expected-run-context-sha256 <sha256> \
  --expected-assembler-sha256 <sha256> \
  --training-output-dir /absolute/path/training-output \
  --training-sidecar-dir /absolute/path/training-sidecar \
  --evaluation-package-dir /absolute/path/evaluation-package \
  --command /absolute/path/command.txt \
  --scheduler /absolute/path/scheduler.json \
  --stdout /absolute/path/stdout.log \
  --stderr /absolute/path/stderr.log \
  --output-dir /absolute/path/retrieved-runs/<run-id>
```

The distinct `--engineering-test-mode` uses a one-submission engineering
campaign and the same source/outer closure checks, but emits
`endpoint_class=bounded_engineering_test` and `analyzer_eligible=false`.
It never invokes or relaxes the production 30,000-update analyzer gate. On
Hopper this is intentionally a two-phase flow: the job publishes only closed
training/evaluation components and exits; after Slurm reports `COMPLETED 0:0`,
the local finalizer creates `scheduler.json` from authoritative terminal
`sacct`, supplies the terminal stdout/stderr, and invokes the assembler. A
running job must never fabricate its own terminal accounting.

Revalidate a published bounded package read-only with the digest printed by
the assembler:

```bash
python3 ued_benchmark/scripts/assemble_matched_run.py \
  --engineering-test-mode --validate-only \
  --campaign-manifest /absolute/path/engineering-campaign.json \
  --expected-campaign-sha256 <sha256> \
  --output-dir /absolute/path/<engineering-run-id> \
  --expected-package-sha256sums-sha256 <sha256>
```

The current upstream/overlay logs are insufficient for analytic-versus-plugin
rank correlation, top-k overlap, Jensen-gap, or replay-mass-by-trial-quartile
diagnostics. The preregistration records the exact missing per-slot telemetry;
the analyzer will not infer it from aggregates or unpickle checkpoints.

Outcome-blind repository preflight:

```bash
python3 ued_benchmark/analysis/preregistered_dev_analysis.py --preflight
```

Sealed analysis (after all jobs have terminal accounting and were retrieved):

```bash
python3 ued_benchmark/analysis/preregistered_dev_analysis.py \
  --campaign-manifest /absolute/path/campaign-manifest.json \
  --expected-campaign-sha256 <sha256> \
  --runs-root /absolute/path/retrieved-runs \
  --output /absolute/path/development-analysis.json
```

The campaign manifest schema and run-package schema are documented in the
module docstring and exercised by synthetic tests. Run:

```bash
python3 -m unittest -v \
  ued_benchmark.analysis.test_preregistered_dev_analysis \
  ued_benchmark.tests.test_assemble_matched_run
```

No current Hopper endpoint is an input to these tests.
