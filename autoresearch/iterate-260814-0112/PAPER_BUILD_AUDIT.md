# Independent paper-build audit

**Frozen candidate:** 2026-08-14  
**Verdict:** GO (`P0=0`, `P1=0`, `P2=0`)  
**Scope:** deterministic local paper artifact only; no experimental endpoint
was inspected.

## Frozen identities

- `reproduce.sh`:
  `bbf214eabd9c14a5067b60e2da00f0b09410102e567ba685c51c7d947d8d3e4b`
- build receipt:
  `07e5b5e37db65dd17792abedd140fa4c3240a50db1d1ea87dd5b0aa9f4fc22b1`
- figure requirements/toolchain:
  `1ceed8ceb0ebb85ad50bea71f8cfada30fdb80abbac9092186537b5677b3bf77` /
  `2f62fc636ef7c38b333f95e081f70320106f9c3fd52b73a95dd27414b58e9e4c`
- compact PDF:
  `36a6c1fb899b3b05477bd7d724899ba4a704b80a90cd0de970626da6d0e3abcb`
- extended PDF:
  `25023b853823133ba7ab38d82fa2d0fec9c328611d19468a47f8f8eec9d16dea`

## Independent checks

- Recomputed the exact Python/package/font record and the Tectonic URL map,
  index, format, and 483-member regular non-symlink tree closure.
- Verified all 20 receipt input rows; together they cover every figure included
  by the compact and extended wrappers.
- From output-free staging, regenerated all 26 script-owned PDF/PNG files
  exactly. This specifically checks that omitted generator outputs cannot pass
  through stale staged files.
- Independently rebuilt the compact and extended PDFs and both logs to their
  frozen byte identities.
- Confirmed 15/25 pages, conclusion on page 8, reproducibility/references on
  page 9, and appendix on page 10.
- Confirmed every disclosed non-fatal warning and the absence of undefined
  citations/references, overfull boxes, fatal errors, and emergency stops.
- Fault injection after two publication renames restored all three prior public
  PDFs exactly, validating the documented rollback behavior.
- Found no account name, email, author identity, absolute workstation path, or
  local timezone leak in PDF metadata/text or the checked-in receipt.
- Shell, JSON, claim-trace, registry-count, TeX-balance, visual-layout, and
  global whitespace/diff checks passed.

The receipt's clean-host bootstrap limitation is accurate: the exact external
tool assets are content-bound but are not vendored. This does not weaken the
deterministic local-build verdict and remains an artifact-release task.
