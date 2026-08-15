# Tie-aware v4 sibling Hopper audit

**Verdict:** LOCAL-ONLY GO; REMOTE/PAPER HOLD  
**P0 in local scope:** none  
**Remote action:** none

## Bundle identity

- bundle ID: `d602ce7854f8f3e99352`
- root manifest:
  `d602ce7854f8f3e99352025b97eed2fde32733c0dd23297d5c28b1051e7aeaf0`
- `BUNDLE_STATE`:
  `7a56b89e0d4fd88e3e591d36c27d1b8ed0a23ee99165038365b608129c799065`
- overlay manifest:
  `cf6009460552ddf8005286f9d131a2be9249f32a76f682121a390491e6b22ada`
- stage script:
  `b36932bd680eea3d48d305ffba514591020a25b5c3d287ef0d4e295348d453ca`

Two retained local bundles are byte-identical and pass their strict root and
overlay manifests. All 15 sibling-tooling files match both bundles. Protected
v3 artifacts and queued job `9367063` are unchanged and explicitly non-reusable
for v4.

## Local gates

- R0 deterministic bundle/import/wrapper test: PASS.
- R2 Frontier one-update CPU test: PASS.
- R3 Frontier and MaxMC Phase-A/B terminal tests: PASS; 60 actual evaluation
  records are sealed inside engineering-only closures.
- Fresh applied-clone tie-aware tests: 22/22 PASS under pinned JAX/JAXlib
  0.4.31 CPU.

## Remaining holds

Do not stage or submit this candidate. A remote engineering release must first
bind and test Hopper's exact export semantics, MIG identity, authoritative
post-terminal `sacct`, fixed Conda/runtime closure, and a manifest-bound
non-test Python 3.10.20 Phase-B environment. Cost100, production, analyzer,
performance/OOD endpoint, and paper-evidence paths remain disabled.

Defense-in-depth before remote release: enforce exact archived receipt keysets,
bind pair-level provenance so both arm campaigns share identical R1/R2
prerequisites, and add realistic Slurm `.batch`/`.extern` resource-row
negatives. These do not weaken the local-only verdict and do not authorize a
remote run.
