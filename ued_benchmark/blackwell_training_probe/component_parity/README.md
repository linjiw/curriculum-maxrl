# Non-updating CPU / RTX 5090 component parity

## Decision

The RTX 5090 training gate remains closed.  The same JAX 0.6.2 code replays
identical initialization, task RNGs, levels, observations, actions, and PPO
minibatch permutation on CPU and GPU, but its recurrent LSTM carry exceeds the
pre-frozen numerical tolerance.  Normalized-advantage elements then diverge,
and first-step Adam normalization amplifies otherwise small, tolerance-passing
near-zero policy-hidden gradients into failing update proposals.

This is an engineering diagnosis, not training or evidence.  No optimizer was
applied, no parameter changed, and no second PPO update, OOD evaluation,
additional seed, throughput measurement, or benchmark endpoint ran.

## Frozen contract

`COMPONENT_PARITY_PROTOCOL.json` was frozen before the GPU capture with
SHA-256:

`0f8c083202a189ec234f32c0e1c15e7c09753892fb05af0d6262b9ff0bf9f1a5`

It retains the earlier tolerances without relaxation:

- CPU: `rtol=1e-6`, `atol=1e-7`.
- GPU: `rtol=5e-4`, `atol=5e-5`.
- Integer, Boolean, task, action, PRNG, shape, dtype, and non-finite contracts
  remain exact.

All lanes load the same frozen JAX 0.4.31 initial checkpoint:

`/data/robotixx/ued_bench/runs/blackwell_training_probe_b7c865/reference-jax0431-cpu-protocol-v6/initial-checkpoint.pkl`

Checkpoint SHA-256:
`4dd07bf02eeb7ec072e4ec72b3aa02180c3ae84284ba20b27174f3dfa9886187`

The only full runner call completes cycle-one warmup.  Robust PLR selects its
fake-update branch, which is checked by unchanged parameter hashes and zero
update counters.  Cycle two is reconstructed only through replay selection,
rollout batch generation, PPO forward/loss/gradient calculation, clipping,
and gradient-transformation proposal.  The script never calls the cycle-two
runner step, agent update, train-state gradient application, or parameter
application.

## Result by stage

The primary comparison is JAX 0.6.2 CPU versus JAX 0.6.2 RTX 5090, so the only
intended variable is backend arithmetic.

| Stage | Result | Key observation |
|---|---|---|
| Initial state | PASS | 92/92 records byte-exact |
| Cycle-one control | FAIL | LSTM carry `[3][0]`; 9 elements fail, max error `1.2597441673278809e-4` |
| Task stream | PASS | 31/31 records byte-exact |
| Rollout observations/rewards/dones | PASS | 4/4 records byte-exact |
| Rollout actions | PASS | Byte-exact |
| Rollout old values/log-probabilities | PASS | Max error `4.029273986816406e-5` |
| Targets/advantages/carry batch | PASS | Max error `8.290261030197144e-5` |
| Minibatch stream | PASS | Discrete permutation and fields exact; floats within gate |
| PPO forward | FAIL | Recurrent carry; 34 elements fail, max error `1.2597441673278809e-4` |
| PPO loss elements | FAIL | Normalized advantage and actor elements; max error `1.4483332633972168e-3` |
| PPO scalar loss terms | PASS | Max error `1.695007085800171e-5` |
| Unclipped gradients | PASS | Max element error `5.101785063743591e-5` within frozen combined tolerance |
| Global norm and clipping | PASS | CPU `0.1379455179`, GPU `0.1379630268`; clip factor `1.0` in both lanes |
| Adam proposal | FAIL | `fc_pi_1` bias and kernel aggregate update magnitudes |

The earliest captured failure is therefore
`forward_or_gemm_recurrent_carry`, not task selection or action sampling.
Cycle-two tasks, actions, and permutations are exact, ruling out sampling as
the source of the earlier one-update parameter discrepancy.

## Why the Adam proposal becomes large

The policy hidden layer is a near-zero-gradient cancellation case:

| Leaf | CPU gradient `abs_sum` | GPU gradient `abs_sum` | CPU update `abs_sum` | GPU update `abs_sum` |
|---|---:|---:|---:|---:|
| `fc_pi_1.bias` | `1.3168447027e-9` | `7.1011764078e-6` | `3.9505010374e-8` | `2.0396194759e-4` |
| `fc_pi_1.kernel` | `2.0190169354e-9` | `1.6724468143e-5` | `6.0570433922e-8` | `4.9697433126e-4` |

Clipping is inactive because both global norms are below `0.5`.  For the zero
Adam state at step one, each proposed update is computed from the captured
post-clipping gradient `g_clip` as
`-0.0003 * g_clip / (abs(g_clip) + 1e-5)`.  A CPU read-only calculation from each saved
`g_clip` matches the captured CPU proposal within `6.67e-16` and the captured
GPU proposal within `8.19e-12`.  The comparator also checks that each captured
`g_clip` equals the raw gradient times the captured clip factor.  An
active-clipping contract test (raw norm `5.0`, clipped norm `0.5`) ensures the
analytic calculation cannot silently fall back to the raw gradient.  Adam is
therefore behaving consistently; it
amplifies an upstream forward/loss/gradient cancellation difference by about
29 times in aggregate.  Relaxing the tolerance would hide, not fix, this
mechanism.

The GPU bias proposal `abs_sum=0.00020396194759086939` also reproduces the
previous one-update checkpoint's bias magnitude (`0.0002039619503193535`) to
roundoff, closing the diagnostic loop without another update.

## Artifacts

Run root:

`/data/robotixx/ued_bench/runs/blackwell_component_parity_0f8c0832`

- JAX 0.4.31 CPU capture SHA:
  `2604739ab7fa32115cca11fd9c469b3a5747bc7353c081993eafdfaa5500dcc0`
- JAX 0.6.2 CPU capture SHA:
  `9aefe688d1630f97799220455fdbae32205874e0b8a0971860d3d9bca0ec6382`
- JAX 0.6.2 RTX 5090 capture SHA:
  `757d5a52d1af8856e4bf7280a90dde0b6966bfc8dc10423d6a8ca0c95351a651`
- Modern CPU comparator self-check SHA:
  `3975c81c9c83dd611c5ffdeb9dcbb71a262d679e621bffb4c46aa5b26133e8fe`
- Primary CPU/GPU comparison SHA:
  `09b4745799e689e62c0b68db900947fdd55a2ab72cb80747721b6067e48ae2d5`

The ancillary JAX 0.4.31/JAX 0.6.2 CPU component comparison fails the stricter
per-element diagnostic gate first at normalized-advantage loss elements.  It
does not replace the already-passed parent one-update CPU compatibility gate;
the primary backend diagnosis intentionally compares JAX 0.6.2 against itself.

## Commands

The modern CPU capture used:

```bash
env -u PYTHONPATH \
  TMPDIR=/data/robotixx/ued_bench/tmp \
  PIP_CACHE_DIR=/data/robotixx/ued_bench/cache/pip \
  PYTHONDONTWRITEBYTECODE=1 \
  JAX_PLATFORMS=cpu \
  JAX_PLATFORM_NAME=cpu \
  JAX_THREEFRY_PARTITIONABLE=false \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  WANDB_MODE=disabled \
  /data/robotixx/ued_bench/envs/jax062-cuda129-probe/bin/python \
  ued_benchmark/blackwell_training_probe/component_parity/capture_component_parity.py \
  --source /data/robotixx/ued_bench/src/minimax-frontier-blackwell-training-jax062-5868d346-d053054 \
  --initial-checkpoint /data/robotixx/ued_bench/runs/blackwell_training_probe_b7c865/reference-jax0431-cpu-protocol-v6/initial-checkpoint.pkl \
  --output /data/robotixx/ued_bench/runs/blackwell_component_parity_0f8c0832/modern-jax062-cpu-v1 \
  --lane modern \
  --backend cpu
```

The GPU form was executed exactly once with `JAX_PLATFORMS=cuda`,
`JAX_PLATFORM_NAME=gpu`, output `modern-jax062-rtx5090-v1`, and otherwise the
same inputs.  Do not rerun it under this protocol.

The read-only comparison can be rerun with:

```bash
env -u PYTHONPATH \
  TMPDIR=/data/robotixx/ued_bench/tmp \
  PIP_CACHE_DIR=/data/robotixx/ued_bench/cache/pip \
  PYTHONDONTWRITEBYTECODE=1 \
  /data/robotixx/ued_bench/envs/jax062-cuda129-probe/bin/python \
  ued_benchmark/blackwell_training_probe/component_parity/compare_component_parity.py \
  --reference /data/robotixx/ued_bench/runs/blackwell_component_parity_0f8c0832/modern-jax062-cpu-v1 \
  --reference-lane modern \
  --candidate /data/robotixx/ued_bench/runs/blackwell_component_parity_0f8c0832/modern-jax062-rtx5090-v1 \
  --backend gpu \
  --output /data/robotixx/ued_bench/runs/blackwell_component_parity_0f8c0832/cpu-vs-rtx5090-comparison-new.json
```

Use a new output name; the comparator refuses overwrite.  Static tests:

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 \
  python3 -m unittest -v \
  ued_benchmark.blackwell_training_probe.component_parity.test_component_parity_contract
```

The complete machine-readable result is in `manifest.json`.
