# Anonymous evidence-release projection

The anonymous release is a derived, evidence-only double-blind supplement, not
a replacement for the frozen source artifacts and not a standalone source
distribution. The generator never edits an input. It copies the declared
public documents, manuscript sources, and every paper-manifest input into
`tmp/anonymous-release/`, except exact manifest inputs declared as unshipped
with a frozen byte count and SHA-256. A complete, independently scanned anonymous source
bundle must be packaged separately before executable-artifact distribution.

The tracked scope is `anonymous_release_scope.json`. It excludes invalid and
aborted experiment branches, TeX/build logs, binary render outputs, the
unshipped Digits replay trees and scratch output. The finalized compact
ProCuRL evidence (lock, protocol/provenance, gate, analysis, portable receipt,
diagnostics, external manifest, post-run tools and tests) is included. Its
1,374,886,097-byte confirmatory raw and 11,453,535-byte development raw are
declared but unshipped with exact SHA-256 values and null public download URIs;
the development raw is the one manifest input intentionally excluded from the
projection. Exact, count-locked transforms remove the
declared machine paths and owner identity and label the projected README's
evidence-only boundary. The generated full receipt records every input/output
digest. The compact tracked receipt records the complete projection-index
digest and, for each changed file, its original/export SHA-256 values,
transformation kind, JSON pointer when applicable, original-value digest,
replacement, and occurrence count.
The receipts also record both unshipped declarations and the exact
manifest-input exclusion, so an audit cannot silently reinterpret the compact
projection as a full raw replay.

Run from the repository root:

```bash
python3 -m curriculum_maxrl.analysis.anonymous_release --audit \
  --check-compact-receipt anonymous_release_receipt.json
python3 -m curriculum_maxrl.analysis.anonymous_release
python3 -m curriculum_maxrl.analysis.anonymous_release --check
```

The policy is fail-closed. It rejects the configured Unix/macOS home, temporary,
root, and volume prefixes; Windows drive-letter user paths and UNC paths;
owner-identity tokens; case-insensitive local-file URLs; duplicate JSON keys;
symlinked inputs, scope configs, compact receipts, or input ancestors; symlinks
substituted into an export; stale occurrence counts or approvals; changed
manifest checksums or manifest-to-unshipped digest bindings; stale or
undeclared manifest-input exclusions; selected excluded artifacts; forbidden receipt/scope
metadata; and unexpected output files. Other generic POSIX locations are not
claimed to be anonymous unless added to the configured detector list. To
refresh after an intentional paper/result change, first review the audit. Then
regenerate and review the compact receipt before updating its tracked bytes.

Projected JSON files with replacements no longer match the original evidence
hashes. That difference is intentional and fully receipted; scientific
verification must continue to use the frozen originals and their existing
locks before projection.
