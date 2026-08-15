# Deterministic paper build receipt

**Built:** 2026-08-14T15:28:48Z  
**Scope:** compact ICLR-2027 manuscript, website copy, and extended research draft  
**Outcome:** PASS. All three tracked PDFs were replaced only after isolated figure
regeneration, two cached TeX builds, exact byte checks, and log validation passed.

This is a review-artifact receipt. It is not paper evidence for the held AMaze,
PLR/ACCEL, robotics, or performance claims.

## Reproduction command

`$PAPER_DATA_ROOT` denotes a user-selected external data/cache root. The final
build used no network access:

```bash
TMPDIR="$PAPER_DATA_ROOT/paper-toolchain/tmp" \
PAPER_FIGURE_PYTHON="$PAPER_DATA_ROOT/paper-toolchain/figures-py311/bin/python" \
XDG_CACHE_HOME="$PAPER_DATA_ROOT/snmr-tools/tectonic-cache" \
TECTONIC_BIN="$PAPER_DATA_ROOT/snmr-tools/tectonic-0.16.9/tectonic" \
PAPER_SOURCE_DATE_EPOCH=1786718220 \
bash reproduce.sh --build
```

`reproduce.sh` SHA-256:
`bbf214eabd9c14a5067b60e2da00f0b09410102e567ba685c51c7d947d8d3e4b`.

## Pinned toolchains

### Figure generation

- CPython `3.11.13`; executable SHA-256
  `4627a60ce761a303bb866244833a914aabab9880b8082fbb0fe8cf35c91ea3ed`.
- `paper/requirements-figures.lock` SHA-256
  `1ceed8ceb0ebb85ad50bea71f8cfada30fdb80abbac9092186537b5677b3bf77`.
- `paper/FIGURE_TOOLCHAIN.json` SHA-256
  `2f62fc636ef7c38b333f95e081f70320106f9c3fd52b73a95dd27414b58e9e4c`.
- Canonical package/font record SHA-256
  `9a05c8bfbd95df8f94aa413e6304b2dbcd709fabb5e006a198c96688004cf614`.
- Matplotlib `3.11.1`, NumPy `2.4.6`; exact remaining package versions
  and the three DejaVu font-file hashes are in `FIGURE_TOOLCHAIN.json`.
- Rendering uses `python -I -B`, a new isolated `MPLCONFIGDIR`, and
  `SOURCE_DATE_EPOCH=1786718220`. Script-owned staged PDFs/PNGs are removed
  before generation, so an omitted output cannot inherit a stale copy. Every
  generated PDF and PNG must then compare byte-for-byte with the tracked
  figure before compilation.

### TeX compilation

- Tectonic `0.16.9`; executable SHA-256
  `397efac4cabf7dfa02f238fe23681215b535ea665e99ba27d123b8bc655b88cb`.
- Bundle identity
  `6ffe055852f8faf66c0acbe1a7fb27f87b869a90bad1204f3bf4d9683f597c7c`.
- URL-to-bundle mapping SHA-256
  `1f94cb6e6893fb09037585fdde65d436f90e2d726175e06363723529f52c880e`.
- Bundle-index SHA-256
  `0fb434b0fa5fdebea7f767ed9c31939c99a780d6f95cd3f540aae55910bb5697`.
- Cached LaTeX format SHA-256
  `a86ffcac335474fb9fae47cd9986b929719dc3ddf29bfb31123ecc1790ef6bbb`.
- Exact 483-member bundle-tree digest
  `e582b51bd80124956fc212c0b1e9da88cb662ccbec2a03a2e3a49c7c31d85a95`.

The tree digest hashes each sorted relative member path, a NUL separator, and
the binary SHA-256 of that regular non-symlink member. Compilation uses
Tectonic `-C --keep-logs` and the fixed source epoch.

## Bound manuscript inputs

| Input | SHA-256 |
|---|---|
| `paper/main_iclr2027.tex` | `d61c80e89a9f7b11305113e641d345bc5cdaf5ed1248a2a2cde1f1fe66f9efea` |
| `paper/body_iclr.tex` | `298a9e432e5bb209ee7a9539c993c6d659e47b2ab952dfe937aa436e5ab9cf5f` |
| `paper/main.tex` | `7233060d2336a4d0ed2eb651b59e46d730b520c84d42ea5055d5e32abaf25bb9` |
| `paper/body.tex` | `444700b3edcb0178300c98d5c22df3cca9ba9c74752d08f4df9b4169c5e8e0ae` |
| `paper/iclr2027_conference.sty` | `797deef41724e93761426ac0cbcca46279a91cc650dd1f0ce76a4f08d2098ea6` |
| `paper/CLAIM_TRACE_ICLR.md` | `f210a8b779950735b99ac64d5796c1c3a98a0ab3c5d2a90b5f7e0fe42253c65f` |
| `paper/results/manifest.json` | `51c5d882be75f8b828af3a6cc54ead8be010c64453b42f18f6e2a5b49e91c0ee` |

## Bound figure inputs

| Input | SHA-256 |
|---|---|
| `paper/figures/fig1_utility.pdf` | `b5777922cb51fa7e82eaded64dcf171b3155245dd6e0f4479d4b56f36b4c500a` |
| `paper/figures/fig_maze_block_contrasts.pdf` | `a90a07bf7a858254b20fdae277ae3cb05cc8a8aa7ae038ad2ddec133f604780f` |
| `paper/figures/fig_countdown_core.pdf` | `56e649d94f2031eb00b0058d81a48241fb8ac366c89f1ed38d572c4b63db7931` |
| `paper/figures/fig4_algorithm.pdf` | `6e5901fb5b42caf15c5e66f4d705ee38701995abbf6f49fd28b8bf130297f1f4` |
| `paper/figures/fig5_channels.pdf` | `f8d095507b6097edbfa5abeb3630b25f39dc04506448f163dda2a4b4667adbab` |
| `paper/figures/fig2_ladder.pdf` | `24a57b89f6a7e60e23f1b10b0a14bd0c60e6d8b7940be25c307982715e0ae58d` |
| `paper/figures/fig8_bands.pdf` | `b1f99f7ef2ec48dceab58ed7fd2fbf41fffbcb53543ef134e4c332898804c055` |
| `paper/figures/fig3_gsm8k.pdf` | `5b0757cd9852f22914f2c059c312c1be07c26b75ff1096427822a4a8d1216bc7` |
| `paper/figures/fig7_sharpening.pdf` | `4b5b999256f0ffa6f9fa5685f7723e0b642f37535cbcee03220e8dcaddf1c963` |
| `paper/figures/fig9_passk.pdf` | `f2a81018c7efac67c2546615364f0ce1b39c28b9a1dc7b2a2e334df0a878d6f6` |
| `paper/figures/fig6_gym.pdf` | `843f45821bc292385249cd32f0f772defcd34371166ef5903c2dc859f2efc2cc` |
| `paper/figures/fig9_ksweep.pdf` | `ac2325c77e5b065d06aeb148408741f5df620b80d32a50efef1d5517ec737cb0` |
| `paper/figures/fig10_jugs_entropy.pdf` | `cdc487f6ab91ba857f73db5f553af5b9dd090b44a4813102ba6df38d73ae9bcf` |

## Published outputs

| Output | Pages | Bytes | SHA-256 |
|---|---:|---:|---|
| `paper/main_iclr.pdf` | 15 | 220,740 | `36a6c1fb899b3b05477bd7d724899ba4a704b80a90cd0de970626da6d0e3abcb` |
| `docs/paper-iclr.pdf` | 15 | 220,740 | `36a6c1fb899b3b05477bd7d724899ba4a704b80a90cd0de970626da6d0e3abcb` |
| `paper/main.pdf` | 25 | 585,608 | `25023b853823133ba7ab38d82fa2d0fec9c328611d19468a47f8f8eec9d16dea` |

Compact log SHA-256:
`d86b8c00f3432da571de6c9bd3eba07ad727cdd26f73ac6099578e53eb8671af`.
Extended log SHA-256:
`9c3af95746d2709a9610336ae0175f26a89e030fb4cfe9c34e9770a47d8a3208`.

## Validation and publication behavior

- Two independent isolated figure builds and two independent cached TeX builds
  were byte-identical.
- The compact conclusion is on page 8; reproducibility/references begin on
  page 9; the appendix begins on page 10.
- Both logs are rejected for undefined references/citations, overfull boxes,
  emergency stops, or fatal errors. None occurred.
- Non-fatal diagnostics are disclosed: each log has nine underfull-box
  diagnostics and two invalid-UTF-8 warnings originating in cached algorithm
  style files; compact has five font-shape substitutions plus the summary
  warning; extended has two `h`-to-`ht` float-placement warnings.
- The isolated Matplotlib render emitted two benign timestamp-normalization
  notices.
- The three PDFs are first copied to same-directory temporary files and hashed.
  Publication uses three atomic per-file renames. If a later rename or final
  hash check fails, the exit trap restores all prior tracked PDFs. Thus the
  set has rollback semantics, while filesystem atomicity is per file rather
  than one cross-directory transaction.
- Final hashes are rechecked after publication; the compact and website copies
  must be identical.
- Static shell syntax, JSON parsing, claim-trace continuity, registry counts,
  manuscript brace/environment balance, and `git diff --check` pass.
- The PDFs contain no author, email, account name, or absolute workstation path
  metadata.

## Portability boundary

The exact Python interpreter, Tectonic executable, and extracted 483-member
cache are checksum-bound external assets, not vendored in this checkout. This
receipt proves deterministic reproduction when those assets are supplied, but
does not yet provide a clean-host bootstrap. An artifact release should publish
an independently retrievable, checksum-bound bootstrap or container for both
toolchains; final compilation must remain network-free.
