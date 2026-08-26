# Anonymous export dry-run receipt — 2026-08-26

This is an artifact-engineering receipt, not scientific evidence. It records a
local dry run only; it does not claim that an anonymous archive has been
uploaded or assigned a DOI.

## Bound inputs

- Source commit: `1145ed059a53`
- Exporter SHA-256:
  `6917dbc8897c0cb58e112e5770af03b143791e0ca17c1b0d975535a5e699dd2a`
- Allowlist SHA-256:
  `6b81d4262421fae57bc0e560eca5859be5d8b8072962b35760160f7e6c0cf8cf`
- Tectonic: pinned 0.16.9 executable and populated bundle cache already bound
  by `reproduce.sh`
- Test/figure interpreter used for this portable run:
  CPython 3.10 with NumPy 1.26.4 and Matplotlib 3.10.8

## Command shape

```bash
EXPORT_PYTHON=/absolute/path/to/python3 \
PYTHON=/absolute/path/to/test-python \
PAPER_FIGURE_PYTHON=/absolute/path/to/figure-python \
TECTONIC_BIN=/absolute/path/to/tectonic \
XDG_CACHE_HOME=/absolute/path/to/populated-cache \
bash scripts/export_anonymous.sh /absolute/new-output-directory
```

The command was run twice into distinct empty output directories from a clean
worktree. Neither output path is part of the receipt or the anonymous archive.

## Result

- Both archives SHA-256:
  `ca90acb63f41cf6a8d958f2be274e26fffc5d3eade145c670565d0588ae3e313`
- Archive size: 4,246,137 bytes
- Snapshot: 384 files, including 22 PDFs; no `.git`, symlinks, raw logs,
  scheduler stdout, interpreter/test caches, or internal anonymity receipts
- Portable reproduction: passed (99 tests passed, one skipped; 17 manifest
  inputs; eight compact figures; two rebuilt manuscripts; preregistered
  verdict spot checks)
- Compact/web PDF SHA-256:
  `37421c77c2d67631b8d0d9b97f33c0991c08b328324a0de6b6039972327497e7`
- Extended PDF SHA-256:
  `dcb2b6f16969a465e5c8259f605a9617283a436faf063a6dd444c12ecd467907`
- Scans passed: personal absolute paths, author-owned repository/site URLs,
  PDF identity metadata, PDF text, and symlinks
- Upload status: `not_uploaded_pi_owned`
- DOI: null

## Transformation audit examples

The exported `ANONYMIZATION_TRANSFORMS.json` contains 20 records and both the
canonical and anonymized SHA-256 for each changed file. Examples:

| File | Canonical SHA-256 | Anonymous SHA-256 | Change |
|---|---|---|---|
| Acrobot analysis | `463fa1a01d95922976f09f75b21f6d8f2c6a8d256081ebedfa4ba968a06f356b` | `6f21c254b953f2c1a6826867e3a81d80bb79a7080a080acd28173aa96900ff11` | two host paths scrubbed |
| P0 analysis | `c1e6dc3ead1ef11db90fa2380999c3f3c45d5bc5c8c8fb234420632bc0d952e9` | `6e7de253bf0ba0681fc14f19f5476524cfbbfb741b78da46be25a9cfd4720653` | 96 host-path occurrences scrubbed |
| Compact manifest | `de66724186790b85ffddee5830cbcb57e3b681c86fbd0705138f6902d2dc4fca` | `cb992f67831a633360dce1f67d260c13815fa378b30cbc89c3e0d738679df2c3` | copied-input checksums rebound |
| Provenance sidecar | `bd01ac19f510c46b3c20772760064d02ab36fe5a44fcc2395c0d87d975941d54` | `5911bc11dcadc4563e13a722fd9b4f61d1adcd23282dd2883f8631c60f03fb6a` | copied-manifest hash/size rebound |

The canonical artifacts were not edited. The exporter changes host paths,
author-owned repository URLs, and dependent checksum bindings only.

## Remaining release action

The PI must rerun this command at artifact freeze, upload the checksum-matched
archive through an anonymous venue, and bind the resulting DOI. Upload and DOI
work are intentionally outside this script.
