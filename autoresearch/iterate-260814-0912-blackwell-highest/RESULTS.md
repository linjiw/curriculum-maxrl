# Results

The one-file patch and fresh-clone reproduction are byte-exact. The patch wraps
only `nn.OptimizedLSTMCell` in `jax.default_matmul_precision('highest')`; GRU,
parameters, initialization, loss, optimizer, environment, and curriculum are
unchanged. Patch SHA-256 is
`a16f4394af0d89289314ab4a11ea43d3334ecba36a22e3c86ed11633d15fb9db`;
the frozen protocol is
`ba0b6fd30de472554d732308017cb8d3c28f7ddef0549631fc5fe907610ec4c3`.

Pre-run validation passed 16 archived minimax tests, 18 non-updating Frontier
formula/buffer tests, and 9 isolated contract/recovery tests.

One zero-optimizer preflight stopped on shared helper API drift and was
explicitly authorized as non-consuming. An isolated compatibility shim retained
the new optimizer-step assertion while adapting the legacy call. The sole
complete CPU candidate then executed two cycles and exactly one PPO/Adam
update, saved both checkpoints, and completed fresh-runner resume. Its base
receipt was written, but the outer wrapper failed immediately afterward while
attempting a second receipt write through a nonexistent module attribute.
The frozen update budget was therefore exhausted; no rerun occurred.

Read-only recovery found all cycle-two scalar values in the raw base receipt.
It loaded the existing checkpoints without calling `experiment.step`. Every
frozen numerical gate passes: 546/546 initial/final leaf aggregates, 24/24
floating statistics, 91/91 exact initial leaf hashes, 949/949 exact gates, and
728/728 non-finite sentinel gates. Maximum aggregate absolute error is
`5.960464477539063e-08`; maximum statistic absolute error is
`7.104790711309761e-07`. The final state has exactly one PPO update, one
gradient update, one Adam application, 64 posterior trials, and no incomplete
or duplicate-new groups. Checkpoint structure is
`ca621b70160c4dd21c94f6cfecc2278dc508a24983ab2c19833c6f39aa1918f0`
with 117 serialized and 91 resumed train-state leaves.

The CPU numerical result is a keep. The bounded-training gate is nevertheless
INCOMPLETE because the provenance-bearing wrapper path did not finish within
the single complete-run budget. RTX 5090 training is therefore discarded for
this iteration: zero GPU updates were attempted. PID 2786996 remains present.

Primary artifacts:

- Manifest: `ued_benchmark/blackwell_training_probe/highest_precision_patch/manifest.json`
- Run: `/data/robotixx/ued_bench/runs/blackwell_highest_precision_ba0b6fd3/modern-jax062-cpu-shim-v1`
- Raw receipt: `98cba2e35bb79ef9037b6286c3605177b2b188a44ab7bb5dff5da75f50edfdf7`
- Exhaustive report: `a168816ee639a25f5ede95d5e17fb9516b4cbb720a934ba448b113f26915ce85`
- Read-only recovery: `824a45896ce919b39cda5d9cf36d50d31ed5a63e92158821571fb70472e44cef`
