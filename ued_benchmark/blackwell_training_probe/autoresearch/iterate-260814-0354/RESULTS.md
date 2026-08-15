# Results

## Kept

- A separate content-addressed `blackwell-training-jax062-v1` overlay replaces
  all 35 removed `jax.tree_map` calls in ten files.  A fresh clone reproduced
  the exact parent and modernization manifests, and no removed call remains.
- The archived upstream suite passes 16/16; the Frontier/project suite passes
  22/22 under an asserted CPU backend.
- Setting `JAX_THREEFRY_PARTITIONABLE=false` restores the source-era PRNG
  stream.  The exact JAX 0.6.2 CPU run starts with 91/91 byte-identical leaves
  and passes one-update parity against JAX 0.4.31, with maximum final aggregate
  absolute error `5.960464477539063e-08`.
- Both CPU lanes preserve exact Frontier counters and exact checkpoint
  structure and round trips.

## Rejected

- The sole RTX 5090 update is not numerically equivalent within the tolerance
  frozen before the run.  Initialization is byte-exact and all control-flow
  counters/checkpoint structure pass, but final
  `['params']['params']['fc_pi_1']['bias']` `abs_sum` differs by
  `0.00020395550519458627`, exceeding GPU `atol=5e-5`, `rtol=5e-4`.
- No second update, OOD evaluation, benchmark run, or evidence run was
  attempted.  The GPU training lane remains closed.

## Receipts

- JAX 0.4.31 CPU reference:
  `1005e3c907c38061f23c46ef8b8b24016818603d4bf42bfd1555afe073b3c8e9`
- JAX 0.6.2 CPU candidate:
  `bb7e16e266c672d268707600598b56ec58f9ae761088d883b74eb81bf820b5c7`
- Read-only GPU recovery:
  `cad634ba29f3455a2cce5af383414f3ff937564487d51e9bb59b36652fd4d446`
