# Anonymous export dry-run receipt — 2026-08-26

This is an artifact-engineering receipt, not scientific evidence. It records a
local dry run only; it does not claim that an anonymous archive has been
uploaded or assigned a DOI.

## Bound inputs

- Source commit: `8cef1a388004`
- Exporter SHA-256:
  `6917dbc8897c0cb58e112e5770af03b143791e0ca17c1b0d975535a5e699dd2a`
- Allowlist SHA-256:
  `6b81d4262421fae57bc0e560eca5859be5d8b8072962b35760160f7e6c0cf8cf`
- Tectonic: pinned 0.16.9 executable and populated bundle cache already bound
  by `reproduce.sh`
- Test interpreter: CPython 3.10. Figure interpreter: pinned CPython 3.11
  with NumPy 2.4.6 and Matplotlib 3.11.1.

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
  `75c2efc8cbcf6d36b52a2b91559fdd78abf8a8fdd673b1938b8d38690ed1824c`
- Archive size: 4,263,540 bytes
- Snapshot: 384 files, including 22 PDFs; no `.git`, symlinks, raw logs,
  scheduler stdout, interpreter/test caches, or internal anonymity receipts
- Portable reproduction: passed (99 tests passed, one skipped; 17 manifest
  inputs; eight compact figures; two rebuilt manuscripts; preregistered
  verdict spot checks)
- Compact/web PDF SHA-256:
  `8e337e232762aa425129371926d7da7d35414b17e37c20ae69070a62957da0d6`
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
| Compact manifest | `3b6bdb62c02a205f231d8438bf96c46177c457fab52fe41ff97c68c3ce071ec4` | `7fd88efa65216a810b904de6ab3a5186a72fd44f890b12a7709105ec775ed6ef` | copied-input checksums rebound |
| Provenance sidecar | `bb7bb87c26a872c1175587d77438fe2cff78aa45d9d7448cb27e7f3ec514a2f1` | `c8986944f602a8ef025bd69189e0c494189f46fbc326545a72c8b79c27ec9ed6` | copied-manifest hash/size rebound |

The canonical artifacts were not edited. The exporter changes host paths,
author-owned repository URLs, and dependent checksum bindings only.

## Remaining release action

The PI must rerun this command at artifact freeze, upload the checksum-matched
archive through an anonymous venue, and bind the resulting DOI. Upload and DOI
work are intentionally outside this script.
