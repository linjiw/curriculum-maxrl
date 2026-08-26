# Anonymous-release audit — 2026-08-26

Status: **PDF green; clean-clone reproduction green; full repository snapshot
red for anonymity. Do not publish the current Git clone or its history as the
anonymous artifact.**

## Verified build surface

- Audited commit: `059c604`.
- A `git clone --no-local` into a fresh temporary directory passed
  `REPRO_MODE=portable bash reproduce.sh --build` without an outside-repository
  scientific-data read.
- The compact and web PDFs were byte-identical at
  `e3d566c40ce211867cd7be4658d4886c4326825083598bc25a7c30b12b38bff6`;
  the extended PDF was
  `f9f387b4e29f1fbb0d4108820f6d6d380c4fdb3e7ea73fa86075bff8607d313c`.
- The compact PDF has 19 total pages. The conclusion is on page 9, the
  uncounted reproducibility and AI-use statements begin on page 9, and
  references begin on page 10.
- `pdfinfo` exposes no Title, Subject, Keywords, or Author value. Its only
  creator fields are `LaTeX with hyperref` and `xdvipdfmx (0.1)`. The seven
  compact figure PDFs expose only Matplotlib creator/backend metadata.
- The build logs contain no undefined reference, undefined citation,
  overfull-box, emergency-stop, or fatal-error diagnostic. Tectonic reports
  underfull boxes and an encoding warning in the vendored ICLR style files;
  neither changes the verified PDF or page boundary.
- Direct links from the site and superseded markdown paper to the author-owned
  repository and Pages site were removed before this audit.

## Repository-snapshot blockers

The fresh clone is reproducible, but it is not anonymous:

- Git history contains four distinct author records and must not accompany a
  double-blind release.
- One immutable Acrobot terminal analysis contains two historical
  identity-bearing absolute-path fields. It remains untouched because silently
  editing a hash-recorded result artifact would violate provenance. Its current
  SHA-256 is
  `463fa1a01d95922976f09f75b21f6d8f2c6a8d256081ebedfa4ba968a06f356b`.
- Historical execution records contain host-specific absolute paths: 63
  tracked files match the local home prefix and 124 match the local data prefix.
  These paths are usually runtime provenance rather than author names, but a
  full-tree release would expose machine/account details and is not portable.
- The configured upstream remote is author-owned. Even a text-clean working
  tree would be deanonymized by publishing the current remote or Git metadata.

## Required release operation

Before external release, the PI must create a **history-free, anonymously
hosted export**, not publish this clone. That export must:

1. use an explicit allowlist for the compact manuscript, manifest inputs,
   analyzers, protocols, result memos, and reproduction code;
2. carry a mechanically produced anonymized copy of the Acrobot analysis that
   changes only the two path fields and records the canonical hash above;
3. omit host-bound logs and working-memory documents not needed by the compact
   claim perimeter;
4. rerun the direct-identity, absolute-path, URL, PDF-metadata, and archive-name
   scans on the exported bytes;
5. pass portable `reproduce.sh --build` from a fresh extraction; and
6. publish the checksum-matched provenance payload only through an anonymous
   archive. The DOI in `PROVENANCE_DEPOSIT.json` must remain null until that
   PI-owned publication succeeds.

This audit deliberately does not mutate the canonical Acrobot result, delete
historical evidence, rewrite Git history, publish an archive, or claim that the
repository is submission-ready.
