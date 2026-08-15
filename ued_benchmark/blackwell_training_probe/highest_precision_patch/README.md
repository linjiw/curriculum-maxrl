# Blackwell highest-precision LSTM patch: bounded result

## Decision

The isolated patch is source- and checkpoint-compatible, and the recovered
JAX 0.6.2 CPU result passes every frozen numerical gate against the JAX 0.4.31
reference. The bounded-training gate nevertheless remains **INCOMPLETE/HOLD**:
the one authorized complete CPU update exhausted the frozen budget before the
outer wrapper completed its provenance-bearing receipt rewrite. No CPU rerun
and no RTX 5090 update were performed after that failure.

| Gate | Result | Basis |
|---|---:|---|
| One-file patch/reproduction | GO | applied and fresh reproduction clones are byte-identical |
| Archived minimax tests | GO | 16/16 CPU |
| Frontier formula/buffer tests | GO | 18/18 CPU, zero optimizer updates |
| Patch/protocol/recovery tests | GO | 9/9 |
| CPU numerical parity | GO | 546/546 aggregates, 24/24 numeric stats, 91/91 exact initial leaf hashes |
| CPU wrapper completion | INCOMPLETE | raw base receipt exists; wrapper rewrite failed after it was written |
| RTX 5090 one-update gate | HOLD | CPU wrapper gate did not complete under the frozen budget |
| Long training/OOD/multiseed/performance/paper evidence | HOLD | not authorized and not executed |

## Patch

The only effective source change is in
`src/minimax/models/common.py`: the `nn.OptimizedLSTMCell` call runs within
`jax.default_matmul_precision('highest')`. The GRU call remains outside the
context. This retains the `OptimizedLSTMCell_0` parameter scope and changes no
parameters, initializers, optimizer, loss, environment, or curriculum logic.

- Applied clone:
  `/data/robotixx/ued_bench/src/minimax-frontier-blackwell-highest-lstm-v1-d053054`
- Fresh reproduction clone:
  `/data/robotixx/ued_bench/src/minimax-frontier-blackwell-highest-lstm-repro-v2-d053054`
- Patch: `a16f4394af0d89289314ab4a11ea43d3334ecba36a22e3c86ed11633d15fb9db`
- Contract: `7d8744ff34d064bd324cdc3d92b972b8050f492ff580edc6e44870bbf4aa969e`
- Applicator: `4fef0fdb4bee747b9794b06832db2ba87345e54e2d21fb1881536521104abd57`
- Applied `common.py`: `b6d56d8cb44a488704d93184b34dade26ed5953ba6ea8012990ea7daa9a9234c`
- Applied overlay manifest: `10f276850036306d9838f44b3266626c8600b69b4a0dcf5757b2bd5468e4d050`
- Frozen one-update protocol: `ba0b6fd30de472554d732308017cb8d3c28f7ddef0549631fc5fe907610ec4c3`

The fresh-clone reproduction command was:

```bash
env -u PYTHONPATH \
  TMPDIR=/data/robotixx/ued_bench/tmp \
  PIP_CACHE_DIR=/data/robotixx/ued_bench/cache/pip \
  python3 \
  ued_benchmark/blackwell_training_probe/highest_precision_patch/apply_highest_precision_patch.py \
  --source /data/robotixx/ued_bench/src/minimax-frontier-blackwell-highest-lstm-repro-v2-d053054
```

`diff -qr` between the applied and reproduction clones returned no output.

## CPU candidate and recovery

The run directory is:

```text
/data/robotixx/ued_bench/runs/blackwell_highest_precision_ba0b6fd3/modern-jax062-cpu-shim-v1
```

The complete candidate used Python 3.10.12, JAX/JAXlib 0.6.2, Flax 0.10.7,
Optax 0.2.5, and TensorFlow Probability 0.25.0. `PYTHONPATH` was cleared,
`JAX_PLATFORMS=cpu` and `JAX_PLATFORM_NAME=cpu` were selected before importing
JAX, source-era Threefry partitioning was retained, and XLA preallocation was
disabled.

The exact complete-candidate command was:

```bash
env -u PYTHONPATH \
  JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu \
  JAX_THREEFRY_PARTITIONABLE=false \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled \
  TMPDIR=/data/robotixx/ued_bench/tmp \
  PIP_CACHE_DIR=/data/robotixx/ued_bench/cache/pip \
  /data/robotixx/ued_bench/envs/jax062-cuda129-probe/bin/python \
  ued_benchmark/blackwell_training_probe/highest_precision_patch/run_highest_precision_one_update.py \
  --source /data/robotixx/ued_bench/src/minimax-frontier-blackwell-highest-lstm-v1-d053054 \
  --output /data/robotixx/ued_bench/runs/blackwell_highest_precision_ba0b6fd3/modern-jax062-cpu-shim-v1 \
  --lane modern --backend cpu \
  --reference-receipt /data/robotixx/ued_bench/runs/blackwell_training_probe_b7c865/reference-jax0431-cpu-protocol-v6/receipt.json \
  --initial-checkpoint /data/robotixx/ued_bench/runs/blackwell_training_probe_b7c865/reference-jax0431-cpu-protocol-v6/initial-checkpoint.pkl
```

The frozen base runner wrote `receipt.json` before the wrapper failed while
trying to call a nonexistent `base._atomic_json`. The writer is fixed for
future use in runner SHA
`dec1c6339f20885ecbae77caecc4c2765e35bc93db217db688b889654191d005`,
but it was not rerun.

The raw receipt preserved both cycle records and all cycle-two scalar values;
there are no missing cycle-two scalar keys. The only missing data are the
outer-wrapper-only shim observation list, GPU-process section, execution-budget
section, and wrapper parity summary. Read-only recovery loaded both checkpoints
on CPU, reproduced their train-state signatures exactly, verified one and only
one Adam application, and exercised pickle and fresh-runner resume without
calling `experiment.step`.

## Exhaustive parity

The corrected exhaustive report stores every individual comparison in
`aggregate_gates`, `statistic_gates`, `exact_gates`, non-finite sentinel gates,
and leaf-hash comparisons. It reports:

- 546/546 aggregate gates: 273 initial and 273 final (`abs_sum`, `squared_l2`,
  and `max_abs` for each of 91 leaves).
- 24/24 floating scalar-statistic gates.
- 949/949 exact structure/counter/checkpoint gates.
- 728/728 non-finite sentinel gates.
- 91/91 exact initial leaf hashes; 24/91 final hashes are byte-identical and
  final hashes are informational, as in the original frozen parity contract.
- Maximum aggregate absolute error:
  `5.960464477539063e-08` at
  `final.['plr_buffer'].scores.max_abs`.
- Maximum aggregate relative error:
  `0.2668467358191151` at
  `final.['params']['params']['fc_pi_1']['bias'].abs_sum`; the absolute error is
  `2.668467358191151e-08`, below the frozen `1e-7` absolute floor.
- Maximum statistic absolute error:
  `7.104790711309761e-07` at `cycle_2.stats.actor_loss`, below the frozen
  `1e-6` absolute tolerance.
- Final counters: two cycles, one PPO update, one gradient update, one Adam
  application, 64 Frontier trials, four filled levels, and zero incomplete or
  duplicate-new groups.
- Checkpoint structure:
  `ca621b70160c4dd21c94f6cfecc2278dc508a24983ab2c19833c6f39aa1918f0`;
  117 serialized leaves and 91 resumed train-state leaves.

The first audit artifact, retained as `aggregate-comparison-invalid-v1.json`,
incorrectly treated non-byte-identical final leaves as failures. The corrected
auditor restores the original contract semantics—exact initial bytes and
tolerance-gated final aggregates—without changing any tolerance. Its SHA is
`b4bb2359f280f990b7b617d7f84976d87c818d43374a1dd400b2a517591ae6f0`.

Key immutable results:

- Raw receipt: `98cba2e35bb79ef9037b6286c3605177b2b188a44ab7bb5dff5da75f50edfdf7`
- Exhaustive corrected report: `a168816ee639a25f5ede95d5e17fb9516b4cbb720a934ba448b113f26915ce85`
- Initial checkpoint: `6e095ca4637d3894717434dde1832dfabf486a3aeb915a87847f985649f98e08`
- Final checkpoint: `d21e282ea7f2e0fa6721a090f9ab570bf6f53b59a76dcf8cf0d5676c595d5151`
- Read-only recovery: `824a45896ce919b39cda5d9cf36d50d31ed5a63e92158821571fb70472e44cef`

## Tests

Archived upstream tests:

```bash
env -u PYTHONPATH \
  JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu \
  JAX_THREEFRY_PARTITIONABLE=false \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONDONTWRITEBYTECODE=1 PYTEST_ADDOPTS='-p no:cacheprovider' \
  TMPDIR=/data/robotixx/ued_bench/tmp \
  PIP_CACHE_DIR=/data/robotixx/ued_bench/cache/pip \
  /data/robotixx/ued_bench/envs/jax062-cuda129-probe/bin/python \
  ued_benchmark/blackwell_training_probe/run_upstream_tests.py \
  --source /data/robotixx/ued_bench/src/minimax-frontier-blackwell-highest-lstm-v1-d053054 \
  --pytest-target /data/robotixx/ued_bench/testdeps/pytest841
```

This passed 16/16. The non-updating Frontier formula/buffer suite passed 18/18.
The final isolated contract suite passed 9/9:

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 \
  TMPDIR=/data/robotixx/ued_bench/tmp \
  PIP_CACHE_DIR=/data/robotixx/ued_bench/cache/pip \
  python3 -m unittest -v \
  ued_benchmark.blackwell_training_probe.highest_precision_patch.test_highest_precision_patch_contract
```

## GPU and evidence boundary

No RTX 5090 update was attempted because the complete CPU wrapper gate is
INCOMPLETE. PID 2786996 remained present after recovery at 7350 MiB. No OOD
evaluation, extra seed, throughput measurement, benchmark endpoint, long
training, performance claim, or paper-evidence run was executed.
