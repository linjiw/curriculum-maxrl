# Highest-precision LSTM-dot closure probe

## Decision

`jax.lax.Precision.HIGHEST` on the two `OptimizedLSTMCell` matrix
multiplications closes the diagnosed CPU/RTX-5090 input-affine and recurrent
carry discrepancies under the existing frozen tolerance.

Default precision reproduces the prior failure.  Highest precision passes
every required recurrent stage, reducing the final-carry maximum error from
`1.2597441673e-4` to `5.9604644775e-8`.

The current training gate remains closed.  This result permits consideration
of a targeted source patch and, only after static and CPU compatibility gates,
a separately authorized one-update parity check.  It is not training,
performance, benchmark, or paper evidence.

## Frozen contract

- Precision protocol SHA-256:
  `0abdb46a7b56986756a31f3d4cc1793af20fc6ca53d2b397720386aab7f5b820`
- Exact reused payload SHA-256:
  `845a34ae40fb762e72b4c6ec569ef16ab6531b241eeaf6cecbc0523059f3bc78`
- Frozen base forward-capture script SHA-256:
  `437e65d445b42d78430c7f84f2e2c4dfe8e2d31ad0973acf031f8831ae40d5a4`

CPU tolerance remains `rtol=1e-6, atol=1e-7`; GPU tolerance remains
`rtol=5e-4, atol=5e-5`.  No threshold was relaxed.  Inputs, parameters,
convolution, reset logic, and RNG provenance are unchanged.  Only the LSTM
input and hidden `jnp.dot` precision differs.

The protocol file's manually entered `frozen_at` value is two minutes late.
Its filesystem timestamp is `04:53:35`, before the sole GPU receipt at
`04:59:20`, and its exact hash is embedded in both captures.  The frozen file
was retained instead of being edited after GPU execution.

## CPU gate

On JAX/JAXlib 0.6.2 CPU:

- canonical default carry equals the decomposed default carry;
- default and highest-precision recurrent tensors are byte-exact;
- final-carry default/highest difference is zero;
- the 77-record comparator self-check passes.

## CPU versus RTX 5090

| Recurrent stage | Default precision | Highest precision |
|---|---:|---:|
| Input affine | FAIL, `1.8253922e-4` | PASS, `8.9406967e-8` |
| Hidden affine | FAIL, `6.7837536e-5` | PASS, `2.9802322e-8` |
| Gate preactivation | FAIL, `2.3016334e-4` | PASS, `8.9406967e-8` |
| Gate activation | FAIL, `1.9334257e-4` | PASS, `1.1920929e-7` |
| Cell state | FAIL, `1.2597442e-4` | PASS, `5.9604645e-8` |
| Hidden state | PASS at existing gate, `6.2644482e-5` | PASS, `2.9802322e-8` |
| Final carry | FAIL, `1.2597442e-4` | PASS, `5.9604645e-8` |

The canonical default model also reproduces the same 34-element cell-carry
failure at maximum error `1.2597442e-4`.  Thus the intervention both preserves
the known failure in its control branch and closes it in its highest-precision
branch.

## Safety and artifacts

Training steps, experiment steps, agent updates, gradients, optimizer
proposals, optimizer applications, parameter mutations, OOD evaluations,
extra seeds, throughput measurements, performance endpoints, and evidence
endpoints were all zero.  Exactly one GPU forward-only capture ran.  PID
2786996 was present before and after at 7350 MiB.

Run root:

`/data/robotixx/ued_bench/runs/blackwell_precision_0abdb46a`

- CPU capture SHA-256:
  `0b7099501fce199c4f61f3e6c77f0da856ee2365f8ebf8a735d1190c92a2e4b2`
- sole RTX 5090 capture SHA-256:
  `62b66e6a6286644ee54e37f9adc0825d871a406b8ce8a9c185176f9f11a33e74`
- CPU self-check SHA-256:
  `60668d9aee765c8464d7afc35ad534535da8502cc3655ef893ce07573ca3cb14`
- primary comparison SHA-256:
  `e1f6034d2ed66492dd2f0df45d93515eea1852d94ff747763e4f10abf8f86f6f`

The complete machine-readable decision is in `manifest.json`.

## Next gate

Implement neither training nor a broad global precision switch directly from
this receipt.  First design a content-addressed compatibility overlay that
changes only the LSTM input/hidden dots while preserving parameter names and
checkpoint structure.  Run static and CPU tests.  A one-update GPU parity gate
may then be reconsidered under a separate execution budget; it is not open now.
