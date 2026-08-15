# Isolated Blackwell training-compatibility probe

## Decision

The JAX 0.6.2 lane is compatible with the bounded Frontier training path on
CPU, but it is **not cleared for RTX 5090 training**.  The CPU lane passes the
frozen JAX 0.4.31 parity protocol after one PPO update.  The sole permitted GPU
update preserves initialization, control-flow counters, Frontier grouping, and
checkpoint structure, but one final parameter aggregate exceeds the tolerance
that was frozen before the GPU run.  The GPU gate therefore fails closed.

This directory records engineering compatibility work only.  It contains no
OOD evaluation, benchmark episode, multi-seed result, throughput result, or
paper evidence.  It does not modify the source-faithful JAX 0.4.31 clone, the
canonical Frontier overlay, or any Hopper file.

## Isolated inputs

- Upstream commit: `d053054c5290a04c1c4cd8b55704d999cad73e30`
- Upstream tree: `b0cace1fc54984e21a842f12d15d0b899e33d270`
- Canonical Frontier contract SHA-256:
  `5868d346ba00e43225424f053ded3dea056b8f76ce4bfdaa1a107a4955cb9000`
- Canonical applied-manifest SHA-256:
  `d929efa2f059a93125e217ec4713ae81670c769d979c67abd2b10efc64268af3`
- Disposable modern clone:
  `/data/robotixx/ued_bench/src/minimax-frontier-blackwell-training-jax062-5868d346-d053054`
- Modern environment: `/data/robotixx/ued_bench/envs/jax062-cuda129-probe`
- Reference environment: `/data/robotixx/ued_bench/envs/minimax-jax0431-cpu`
- External pytest target: `/data/robotixx/ued_bench/testdeps/pytest841`

The modern lane uses Python 3.10.12, JAX/JAXlib 0.6.2, Flax 0.10.7,
Optax 0.2.5, TensorFlow Probability 0.25.0, and NumPy 2.2.6.  The reference
lane uses Python 3.10.19, JAX/JAXlib 0.4.31, Flax 0.8.5, Optax 0.2.3, and
TensorFlow Probability 0.23.0.  The full modern freeze remains in
`../blackwell_probe/environment.freeze.txt`; pytest was installed to a
separate `/data` target and is frozen in `pytest-target.freeze.txt`.

Every package or resolver action must set both `TMPDIR` and `PIP_CACHE_DIR` to
the `/data/robotixx/ued_bench` locations.  Every isolated check clears the
shell-injected `PYTHONPATH`.

## Minimal modernization overlay

`MODERNIZATION_CONTRACT.json` content-addresses a mechanical replacement of
all 35 remaining uses of the removed `jax.tree_map` alias with
`jax.tree_util.tree_map`.  Ten files change; no algorithm, shape, default, or
Frontier contract is changed.  One trailing space on the mechanically touched
line in `agent_pop.py` is removed and declared in the contract.

`apply_blackwell_training_overlay.py` accepts only the pinned upstream tree
with the exact canonical Frontier applied manifest.  It rejects partial or
mixed states, checks every pre/post file digest, writes a separate
`.blackwell_training_overlay.json`, and never edits the parent overlay
manifest.  A fresh-clone reproduction passed with these final hashes:

- Modernization contract:
  `b7c865e007634c5a20e2b942ff98f24d6ac9ff624d5b17b62e5e9fa2124e5c00`
- Applied modernization manifest:
  `ea5fb73c0072cd95829630344e559f02a83f65b0f8b479845ef4dff8921ff65c`
- Applicator:
  `901e487b655c74a6b0a5ecc780fc15ccdf7167755aeb8cb8f4d3002b6a3989b9`

Check the prepared clone without modifying it:

```bash
/data/robotixx/ued_bench/envs/jax062-cuda129-probe/bin/python \
  ued_benchmark/blackwell_training_probe/apply_blackwell_training_overlay.py \
  --target /data/robotixx/ued_bench/src/minimax-frontier-blackwell-training-jax062-5868d346-d053054 \
  --check
```

Expected status: `applied`, 10 files, 35 replacements.  A recursive source
scan finds zero remaining `jax.tree_map` calls, and `git diff --check` passes.

## CPU tests and parity gate

JAX 0.6 changed the default Threefry partitioning mode.  With the new default,
seeded initialization is not comparable to the source-era lane.  The frozen
protocol therefore requires `JAX_THREEFRY_PARTITIONABLE=false`; direct
split/uniform/categorical probes then reproduce JAX 0.4.31.  CPU isolation
also requires both `JAX_PLATFORMS=cpu` and `JAX_PLATFORM_NAME=cpu` before JAX
is imported.  The scripts assert the resulting backend and device.

Final focused results under those settings:

- Archived upstream minimax tests: 16/16 passed.
- Frontier/project tests: 22/22 passed.  TensorFlow Probability emitted one
  deprecation warning for `jax.interpreters.xla.pytype_aval_mappings`.
- Exact 4x8 Frontier CPU protocol: passed.
- Native seeded initialization: 91/91 train-state leaves byte-exact.
- Final state: 24/91 leaf hashes exact; maximum aggregate absolute difference
  `5.960464477539063e-08`, within the frozen `rtol=1e-6`, `atol=1e-7` leaf
  gate.  All scalar statistics pass `rtol=1e-6`, `atol=1e-6`.
- Checkpoints: 117 serialized leaves, 91 resumed train-state leaves, exact
  pickle and fresh-runner resume round trips, common structure SHA-256
  `ca621b70160c4dd21c94f6cfecc2278dc508a24983ab2c19833c6f39aa1918f0`.

The frozen schedule performs two full outer cycles: warmup/insertion followed
by forced replay, with four parallel levels, eight rollout streams per level,
Frontier `N=8`, rollout horizon two, and exactly one PPO gradient update (128
transitions total).  It executes no periodic or OOD evaluation.  Counters are
exact: 32 posterior trials and zero updates after cycle one; 64 trials, one
update, one gradient update, four filled levels, zero incomplete groups, and
zero duplicate new groups after cycle two.

Reference receipt:

`/data/robotixx/ued_bench/runs/blackwell_training_probe_b7c865/reference-jax0431-cpu-protocol-v6/receipt.json`

SHA-256:
`1005e3c907c38061f23c46ef8b8b24016818603d4bf42bfd1555afe073b3c8e9`

Modern CPU receipt:

`/data/robotixx/ued_bench/runs/blackwell_training_probe_b7c865/modern-jax062-cpu-protocol-v6/receipt.json`

SHA-256:
`bb7e16e266c672d268707600598b56ec58f9ae761088d883b74eb81bf820b5c7`

Run the final CPU-isolated archived tests with:

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
  ued_benchmark/blackwell_training_probe/run_upstream_tests.py \
  --source /data/robotixx/ued_bench/src/minimax-frontier-blackwell-training-jax062-5868d346-d053054 \
  --pytest-target /data/robotixx/ued_bench/testdeps/pytest841
```

`run_parity_one_update.py` records and validates the same environment,
source, schedule, counter, checkpoint, and numerical invariants.  Its SHA-256
is `078ac8deb7dc79b2c3fa8ede65a7b818afae7a70f01221f1204ec170339358f2`;
the immutable parity protocol SHA-256 is
`bcd1ba38435b43312e8f4559fad4efdae96169a9a806cb41d551dc76bb8420aa`.

## Sole RTX 5090 update: failed closed

Only one GPU PPO update was allowed and executed.  Initialization loaded from
the JAX 0.4.31 reference is 91/91 leaves byte-exact.  The final checkpoint has
the exact expected structure and counters: two cycles, one PPO update, one
gradient update, 64 posterior trials, four filled levels, and no incomplete or
duplicate groups.  The process saved both checkpoints but did not emit its
normal receipt after XLA Blackwell dot-autotuning warnings.  No retry or second
GPU update was launched.

A CPU-only, read-only recovery loaded those checkpoints without calling
`experiment.step`.  It found exactly one failed aggregate:

```text
path:            ['params']['params']['fc_pi_1']['bias']
metric:          abs_sum
reference:       6.445124767218147e-09
RTX 5090:        0.0002039619503193535
absolute error:  0.00020395550519458627
frozen gate:     rtol=0.0005, atol=0.00005
```

The other aggregates pass; 21/91 final leaves are byte-exact.  This localized
failure is enough to reject GPU numerical parity.  It must be diagnosed with
non-updating forward/loss/gradient component probes before any request for a
new optimizer update.  It must not be hidden by loosening the frozen gate.

GPU run directory:

`/data/robotixx/ued_bench/runs/blackwell_training_probe_b7c865/modern-jax062-rtx5090-one-update-v1`

Recovery receipt SHA-256:
`cad634ba29f3455a2cce5af383414f3ff937564487d51e9bb59b36652fd4d446`

The pre-existing GPU process PID 2786996 was preserved.  All JAX commands set
`XLA_PYTHON_CLIENT_PREALLOCATE=false`.

## Safe next work

The next bounded iteration should compare CPU and GPU, without an optimizer
update, at four cut points: rollout batch, PPO loss terms, gradients before
clipping, and gradients/updates after clipping.  Start with the policy hidden
layer bias path and determine whether the difference originates in a
Blackwell GEMM/reduction kernel, action sampling, or optimizer arithmetic.
Only a new predeclared protocol and explicit authorization should open another
GPU-update budget.  Longer training, OOD evaluation, multi-seed comparison,
and benchmark evidence remain prohibited until that gate passes.

Run the local static contract tests with:

```bash
python3 -m unittest -v \
  ued_benchmark.blackwell_training_probe.test_training_probe_contract
```

The complete machine-readable outcome is in `manifest.json`.
