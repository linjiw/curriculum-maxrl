# Isolated RTX 5090 / JAX 0.6.2 AMaze probe

## Outcome

The pinned `minimax` source plus Frontier overlay imports on JAX/JAXlib 0.6.2,
and one AMaze reset and one AMaze step compile and execute on the RTX 5090.
The only source change needed for that narrow path is a two-line replacement of
the removed `jax.tree_map` alias with `jax.tree_util.tree_map` in
`src/minimax/envs/environment.py`.

This is an engineering compatibility result. It includes no training, no
benchmark episode, and no paper evidence. The source-faithful JAX 0.4.31 lane,
the canonical Frontier overlay, and all Hopper files are unchanged.

## Pinned inputs

- Upstream commit: `d053054c5290a04c1c4cd8b55704d999cad73e30`
- Upstream tree: `b0cace1fc54984e21a842f12d15d0b899e33d270`
- Frontier contract: `5868d346ba00e43225424f053ded3dea056b8f76ce4bfdaa1a107a4955cb9000`
- Disposable modern clone:
  `/data/robotixx/ued_bench/src/minimax-frontier-blackwell-jax062-5868d346-d053054`
- Isolated environment: `/data/robotixx/ued_bench/envs/jax062-cuda129-probe`
- GPU: NVIDIA GeForce RTX 5090; driver 590.48.01
- Python 3.10.12; JAX/JAXlib/CUDA plugin/PJRT 0.6.2; NumPy 2.2.6
- CUDA runtime wheel 12.9.79; NVCC wheel 12.9.86; cuDNN 9.24.0.43

The full resolved environment is in `environment.freeze.txt`. The minimum
direct additions used to make the eager `minimax` import succeed are in
`requirements.direct.txt`. The probe environment passes `pip check`.

The upstream package metadata cannot be installed unchanged in this lane:
`minimax` declares `numpy>=1.25,<1.26`, while JAX 0.6.2 requires NumPy at least
1.26. This probe therefore uses a separately labeled NumPy 2.2.6 / pandas
2.3.0 environment and source loading, rather than weakening the faithful lane.

## Failure ladder

1. The bare JAX 0.6.2 environment discovered the 5090 and compiled a tiny JAX
   operation, but `minimax` import from its `src` directory first failed with
   `ModuleNotFoundError: No module named 'chex'`.
2. After installing the pinned compatible import dependencies, `import minimax`
   passed with `PYTHONPATH` unset.
3. The first corrected AMaze reset compiled on the GPU, while the first step
   failed at `environment.py:85` with:
   `AttributeError: jax.tree_map was removed in JAX v0.6.0`.
4. Applying `minimax-jax062.patch` changed exactly the two auto-reset calls.
   The bounded probe then passed: 5x5x3 observations, state time 1, GPU backend,
   1.050640 seconds for first reset compile/execute and 1.344891 seconds for
   first step compile/execute. These timings are diagnostics, not performance
   measurements.

## Reproduce

All resolver and temporary data must stay on `/data` because the root
filesystem is nearly full:

```bash
mkdir -p /data/robotixx/ued_bench/tmp /data/robotixx/ued_bench/cache/pip

env -u PYTHONPATH \
  TMPDIR=/data/robotixx/ued_bench/tmp \
  PIP_CACHE_DIR=/data/robotixx/ued_bench/cache/pip \
  /data/robotixx/ued_bench/envs/jax062-cuda129-probe/bin/python \
  -m pip check

env -u PYTHONPATH \
  TMPDIR=/data/robotixx/ued_bench/tmp \
  PIP_CACHE_DIR=/data/robotixx/ued_bench/cache/pip \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  WANDB_MODE=disabled \
  /data/robotixx/ued_bench/envs/jax062-cuda129-probe/bin/python \
  ued_benchmark/blackwell_probe/run_amaze_jit_probe.py \
  --source /data/robotixx/ued_bench/src/minimax-frontier-blackwell-jax062-5868d346-d053054 \
  --manifest ued_benchmark/blackwell_probe/manifest.json
```

Expected prefix: `AMAZE_JIT_OK`. The script fails closed if `PYTHONPATH` is
set, XLA preallocation is not disabled, the source/overlay/file hashes differ,
the backend is not GPU, or the reset/step outputs violate the AMaze contract.

Run the static contract tests with:

```bash
python3 -m unittest -v ued_benchmark.blackwell_probe.test_probe_contract
```

## Boundary before training

The original tree contained 35 uses of the removed `jax.tree_map` alias. This
AMaze-only patch fixes two; 33 remain across PPO, batch environments, models,
DR/PLR runners, mutation, and replay utilities. Full Frontier or baseline
training on JAX 0.6.2 is therefore a separate no-go until all remaining calls
are ported in a new content-addressed modernization overlay and its behavior is
checked against JAX 0.4.31. PPO/TFP execution, sharding, checkpoints, full
rollouts, numerical parity, and throughput are also unvalidated.
