# Anonymous-release audit — 2026-08-26

Status: **PDF green; clean-clone reproduction green; allowlisted anonymous
export green. The full Git clone and its history remain red for anonymous
release.**

## Verified release surface

- The export implementation is `scripts/export_anonymous.sh`; its source
  allowlist is `scripts/anonymous_allowlist.txt`. The verified source commit is
  `8cef1a388004`.
- Two independent exports from that clean commit produced the same archive
  SHA-256:
  `75c2efc8cbcf6d36b52a2b91559fdd78abf8a8fdd673b1938b8d38690ed1824c`.
  The archive is 4,263,540 bytes and contains a 384-file, history-free
  snapshot with 22 PDFs.
- Each extracted snapshot passed portable `reproduce.sh --build`: 99 focused
  tests passed and one skipped, 17 manifest inputs verified, all eight compact
  figures regenerated, both manuscript PDFs rebuilt, and every preregistered
  verdict spot check passed.
- The compact and web PDFs are byte-identical at
  `8e337e232762aa425129371926d7da7d35414b17e37c20ae69070a62957da0d6`;
  the extended PDF is
  `dcb2b6f16969a465e5c8259f605a9617283a436faf063a6dd444c12ecd467907`.
- The compact PDF has 20 total pages. The conclusion remains on page 9;
  references begin on page 10. PDF metadata exposes no Author, Title,
  Subject, or Keywords value.
- The exporter removed Git metadata, internal anonymity records, working
  plans, raw execution logs, scheduler stdout, and generated caches. It then
  passed scans for personal absolute paths, author-owned repository/site URLs,
  PDF metadata, PDF text, and symlinks.
- Twenty text/checksum transformations are recorded in the exported
  `ANONYMIZATION_TRANSFORMS.json`. The immutable Acrobot source remains
  unchanged at
  `463fa1a01d95922976f09f75b21f6d8f2c6a8d256081ebedfa4ba968a06f356b`;
  its exported copy changes only two host paths and hashes to
  `6f21c254b953f2c1a6826867e3a81d80bb79a7080a080acd28173aa96900ff11`.
  The export similarly records every other scrub and rebinds the copied
  manifest/provenance checksums to the copied bytes. Scientific values are not
  transformed.

The complete command, hashes, transformation examples, and remaining PI-owned
steps are recorded in `paper/ANONYMOUS_EXPORT_DRY_RUN_2026-08-26.md`.

## Why the full repository is still not an anonymous artifact

- Git history contains multiple author records.
- Historical execution records contain host-specific paths and working-memory
  material outside the compact claim perimeter.
- The configured upstream remote is author-owned. Publishing this clone, its
  `.git` directory, or its remote would disclose identity even though the
  manuscript itself is anonymous.

## Remaining PI-owned release operation

At the artifact freeze, rerun the exporter from a clean release commit, retain
the resulting archive checksum, upload only that history-free archive through
an anonymous host, and bind the returned DOI in the external deposit record.
The DOI remains null and upload status remains `not_uploaded_pi_owned` until
that publication succeeds. Do not publish the current clone or rewrite the
canonical result artifacts.
