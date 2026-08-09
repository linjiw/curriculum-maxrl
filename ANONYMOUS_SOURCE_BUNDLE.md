# Anonymous standalone source bundle

This release form is a deterministic, executable companion to the anonymous
evidence projection.  The frozen scope contains exactly 244
allowlisted source/evidence
files and one generated `ANONYMIZATION_RECEIPT.json`.  It does not contain the
Git repository, machine-local paths, generated figures or PDFs, the multi-GB
Digits replay payload, invalid branches, or the external ProCuRL raw ledgers.

From a freshly extracted bundle, run:

```bash
bash reproduce.sh
bash reproduce.sh --build
```

The first command runs the CPU tests, frozen endpoint derivations, original
scientific-hash checks, and figure regeneration.  The second also builds both
LaTeX wrappers and refreshes the two documentation PDFs.  `uv`, a TeX
installation, and the Python packages named by `reproduce.sh` are required.

## Why there are two hashes for some files

Anonymization changes only explicitly approved path or identity fields.  For
every selected file, the receipt records both the SHA-256 of the original
research artifact and the SHA-256 of the exported bytes.  Frozen source locks,
authorizations, and paper manifests continue to cite the original SHA-256;
bundle-aware verification first validates the exported bytes, then follows the
receipt back to that original digest.  The exporter never rewrites a projected
file and calls it the original.

Two historical Acrobot V1 pre-audit files are named by the retained V2 source
lock but deliberately not shipped because their filenames mark them as aborted
branches.  Their original byte counts and hashes appear as explicit omission
receipts.  The portable check requires those two exact omission records; they
are provenance witnesses, not evidentiary outcomes.

The retained ProCuRL compact closure contains 21 files: its sealed lock,
protocol and provenance; results, passing development gate, confirmatory
analysis, portable verification receipt, external-raw manifest and descriptive
diagnostics; and the runner, analyzer, builders, verifiers and tests needed to
audit those objects.  The invalid pre-gate incident named by the lock is a third
hash-bound omission.  The 1,374,886,097-byte confirmatory ledger and
11,453,535-byte development ledger are separate never-read declarations.  Both
have exact size/SHA-256 bindings and a null public download URI.  The compact
check validates all 21 retained files, every locked original-source hash, the
320-entry external index and 332 registry rows.  It does not claim to replay
the absent raw ledgers.  Two locked verifier tests that open/tamper with the
omitted invalid-incident file are shipped for auditability but deselected in
the compact reproduction; its exact original hash is checked through the
omission receipt instead.

## Verification boundary

The bundle establishes exact file inventory, no symlinks, no duplicate JSON
keys, no unapproved identity/path content, deterministic derivations from the
retained evidence, and reproducible paper rendering.  It does not rerun the
large training executions or supply omitted replay payloads.  In particular,
full ProCuRL raw validation, confirmatory reanalysis, diagnostic regeneration,
and development-ledger row regeneration require the two exact external raw
objects; the compact release only verifies their frozen bindings.  A successful
source build prints the selected-file count, group counts, archive SHA-256, and
zero-leak status.  The definitive archive and receipt are regenerated only
after the manuscript/PDF freeze.  The tracked scope is a builder input and is intentionally
not copied into the anonymous archive; all information required to verify an
extracted archive is embedded in its generated receipt.
