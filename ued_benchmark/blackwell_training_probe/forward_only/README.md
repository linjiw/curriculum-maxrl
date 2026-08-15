# Forward-only RTX 5090 recurrent localization

## Decision

The RTX 5090 training gate remains closed.  With the exact same frozen JAX
0.6.2 parameters, two-step minibatch, initial carry, reset mask, and RNG
provenance, convolution and feature construction pass the existing GPU
tolerance.  The first failure is the `OptimizedLSTMCell` concatenated input
matrix multiplication at time index zero.

This is a non-updating engineering diagnostic.  It ran no experiment or agent
step, gradient, optimizer proposal, optimizer application, parameter mutation,
OOD evaluation, additional seed, throughput test, performance endpoint, or
paper-evidence endpoint.

## Frozen inputs and protocol

The exact payload was frozen before GPU execution:

- `FORWARD_PAYLOAD.json` SHA-256:
  `845a34ae40fb762e72b4c6ec569ef16ab6531b241eeaf6cecbc0523059f3bc78`
- `FORWARD_ONLY_PROTOCOL.json` SHA-256:
  `024239a6b659097198a6d902b1bb63698849d38e340ac033fa21537b0e5888ce`
- frozen source CPU component capture SHA-256:
  `9aefe688d1630f97799220455fdbae32205874e0b8a0971860d3d9bca0ec6382`
- original initial checkpoint SHA-256:
  `4dd07bf02eeb7ec072e4ec72b3aa02180c3ae84284ba20b27174f3dfa9886187`
- reconstructed parameter-tree SHA-256:
  `2343e57072e914aff88c55a31faea54fe2a5b4eba21c5a459a66545720b9d954`

The retained tolerances are `rtol=1e-6, atol=1e-7` for CPU gates and
`rtol=5e-4, atol=5e-5` for CPU/GPU comparison.  They were not relaxed.

The CPU capture first reproduced the frozen canonical PPO model output and
then verified that the decomposed CNN/LSTM equations reproduce the canonical
final carry.  Its 65-record comparator self-check passes.

## Localization

| Operation | Result | Maximum absolute CPU/GPU error |
|---|---:|---:|
| Exact input payload | PASS | `0` |
| Convolution preactivation | PASS | `8.9406967e-8` |
| ReLU/visual flatten | PASS | `5.9604645e-8` |
| Scalar embedding | PASS | `0` |
| Concatenated features | PASS | `5.9604645e-8` |
| LSTM input affine, all gates | **FAIL** | `1.8253922e-4` |
| LSTM hidden affine | downstream failure | `6.7837536e-5` |
| Gate preactivation | downstream failure | `2.3016334e-4` |
| Cell state | downstream failure | `1.2597442e-4` |
| Canonical final carry | downstream failure | `1.2597442e-4` |

The earliest failing record is the forget-gate slice of
`jnp.dot(step_features, concatenated_input_kernel)`, shape `[2,32,16]`, at
index `[time=0, batch=0, hidden=2]`.  It has 195 element failures and maximum
error `1.7858297e-4`.  The other three input-gate affine slices also fail at
time zero.

A read-only counterfactual propagates the captured feature difference through
the exact frozen kernels in float64.  The upstream feature difference can
account for at most `4.7408848e-8` at any gate, while observed affine errors
are `1.4314055e-4` to `1.8253922e-4`, a factor of about 3,617 to 4,437 larger.
The evidence therefore points to default-precision backend GEMM arithmetic,
not convolution, feature layout, reset logic, sampling, or optimizer
semantics.  This inference does not identify a specific GPU kernel or precision
mode without a separately preregistered precision probe.

## Artifacts

Run root:

`/data/robotixx/ued_bench/runs/blackwell_forward_only_024239a6`

- CPU capture SHA-256:
  `ef1313d15f1fa1587d878d500732b1a59f5e07a3e280a939266d8a485229445c`
- sole RTX 5090 capture SHA-256:
  `39d2a4eba7773568f88dbc93d735ce9924fc9bb8ee5f22efbca277e2a6261b25`
- CPU comparator self-check SHA-256:
  `ef80527da7211c7d7eec2e37a15a1e111ba3f988294796df323084d61d321c65`
- primary CPU/GPU comparison SHA-256:
  `1716d6ccbecaa57bf7babb4028e48bc4a8914efd6546afd847158e735d1e2927`

The complete machine-readable result is `manifest.json`.

## Reproduction commands

CPU capture:

```bash
env -u PYTHONPATH \
  TMPDIR=/data/robotixx/ued_bench/tmp \
  PIP_CACHE_DIR=/data/robotixx/ued_bench/cache/pip \
  PYTHONDONTWRITEBYTECODE=1 \
  JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu \
  JAX_THREEFRY_PARTITIONABLE=false \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /data/robotixx/ued_bench/envs/jax062-cuda129-probe/bin/python \
  ued_benchmark/blackwell_training_probe/forward_only/capture_forward_only.py \
  --source /data/robotixx/ued_bench/src/minimax-frontier-blackwell-training-jax062-5868d346-d053054 \
  --output /data/robotixx/ued_bench/runs/blackwell_forward_only_024239a6/modern-jax062-cpu-new \
  --backend cpu
```

The GPU form was executed once with `JAX_PLATFORMS=cuda`,
`JAX_PLATFORM_NAME=gpu`, and output `modern-jax062-rtx5090-v1`.  Do not rerun
it under this protocol.

Static tests:

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 \
  python3 -m unittest -v \
  ued_benchmark.blackwell_training_probe.forward_only.test_forward_only_contract
```

## Next gate

If authorized separately, freeze the same payload under a protocol that tests
`jax.default_matmul_precision("highest")` or explicit highest precision for
both LSTM dots.  Compare forward tensors first; do not resume training unless
that new gate passes and a distinct one-update parity budget is approved.
