# FrontierRL × `minimax` AMaze benchmark overlay

This directory contains an isolated, reproducible extension of
[`facebookresearch/minimax`](https://github.com/facebookresearch/minimax) at
commit `d053054c5290a04c1c4cd8b55704d999cad73e30`. The source-faithful clone is
never edited. The historical v3 overlay changes only the PLR score and the
state required to estimate it. The opt-in v4 overlay additionally makes exact
score-tie replay probabilities permutation-invariant while leaving PPO, buffer
admission, with-replacement sampling, and the PLR staleness-mixture semantics
unchanged.

## Version boundary and tie-aware rank contract

`OVERLAY_CONTRACT.json` and `scripts/apply_minimax_overlay.py` are the immutable
v3 reconstruction artifacts. `OVERLAY_CONTRACT_V4.json` and
`scripts/apply_minimax_overlay_v4.py` are separate v4 artifacts; their exact
lineage is recorded in `OVERLAY_LINEAGE.json`.

V4 preserves source stable-rank behavior by default. When
`plr_tie_aware_score_ranks=true`, only filled slots participate in ranking.
For an exact-score tie occupying one-indexed ranks `l..r`, every member receives
the mean of the already temperature-transformed masses `j^(-1/temp)` over that
block. Thus the block keeps the same score-component mass mathematically; the
fixed-order float32 implementation is checked within a frozen numeric tolerance.
Tie equality itself is exact (`+0.0` and `-0.0` tie), with no epsilon jitter.
Replay draws remain sequential and with replacement; `force_unique` only
deduplicates later buffer updates and never resamples a replay batch.

Tie mode normalizes score mass after sorting the masses into canonical ascending
value order, so its final score and replay probabilities are bitwise equivariant
to buffer-slot permutations. The opt-out path retains upstream's slot-order
float32 sum exactly. Consequently, for all-distinct scores, raw singleton rank
masses remain bit-identical to upstream while normalized stable-vs-tie
probabilities are mathematically equivalent within the frozen absolute float32
tolerance `5e-7`; universal normalized bit parity is impossible without changing
the upstream default reduction.

The v4 Frontier and group-matched MaxMC configs explicitly enable this policy:

- `maze_frontier_exact_grouped_n8_tie_aware_v4.json`
- `maze_maxmc_group_matched_4x8_b500_tie_aware_v4.json`

`maze_maxmc_upstream_official_reference_32x1_b4000.json` contains only pristine
upstream arguments and must run from the unmodified pinned source. The separate
`maze_maxmc_v4_stable_rank_compat_32x1_b4000.json` is a behavior-compatibility
arm, not the source-faithful reference. The v2 tie-aware protocol and terminal
driver are DRAFT engineering infrastructure and explicitly cannot create paper
or matched-development evidence.

## Score contract

For a level with Beta posterior

```text
a = successes + prior_alpha
b = trials - successes + prior_beta,
```

the normalized coefficient activity is

```text
u_N(p) = 1 - (1 - p)^N - p.
```

The default `expected_activity` mode uses the deterministic Bayesian score

```text
E[u_N(p)] = 1 - (b)_N / (a + b)_N - a / (a + b).
```

This is preferable to silently plugging in the posterior mean: because
`u_N` is concave for `N >= 2`, `E[u_N(p)] <= u_N(E[p])`. The explicit
`mean_plugin` mode is retained as an ablation. Both paths evaluate the exact
declared formula and clip float32 roundoff to `[0, 1]`.

Each evaluation stream contributes at most one Bernoulli observation per
rollout. A stream is observed if any terminal transition occurs, and succeeds
if any terminal reward is greater than `frontier_success_threshold` (zero for
AMaze). Multiple early-success/reset episodes in one stream still count once;
an incomplete stream counts neither way. Counts are stored per replay-buffer
slot, reset on eviction, and included in checkpoints.

## Group-size contract

`frontier_n_rollouts` is the estimator budget `N`; `n_eval` is the number of
evaluation streams for each sampled level. Strict mode rejects a mismatch.
The three supplied arms are intentionally unmistakable:

- `maze_frontier_exact_grouped_n8.json`: primary score-isolation arm with
  `n_parallel=4`, `n_eval=8`, `N=8`, strict matching, and buffer size 500. It
  retains 32 simultaneous streams and reaches its 50% replay warmup after the
  same 2,000 evaluated streams as the official 32x1, buffer-4000 baseline.
- `maze_maxmc_group_matched_4x8_b500.json`: group-, buffer-, and
  stream-matched MaxMC control for the primary Frontier arm.
- `maze_frontier_posterior_bridge_n8_neval1.json`: upstream-shaped
  `n_parallel=32`, `n_eval=1`, `N=8`, strict matching disabled. This is a
  low-cost, counterfactual posterior bridge ablation—not current-batch
  coefficient activity.

The grouped arm visits four distinct levels per rollout rather than 32. The
group-matched MaxMC arm isolates the score under that layout. The official
32x1, buffer-4000 robust-PLR arm remains the source-faithful reference, and the
bridge is only a compute-shaped ablation; equal stream-steps alone do not
control level diversity.

Logs expose `frontier_n_rollouts`, `frontier_n_eval`,
`frontier_group_size_match`, posterior-weighted probability/trials, and total
success/trial counts. Strict cells with fewer than `n_eval` observed streams
are rejected and counted. Duplicate buffered replay occurrences accumulate all
evidence; duplicate new groups retain upstream unique-buffer behavior but are
explicitly rejected and counted rather than disappearing silently. Mixed
existing/new batches process existing identities before any new eviction, so
posterior evidence cannot cross-contaminate replay slots. All
patched PLR experiment IDs encode the overlay version/contract hash;
Frontier IDs additionally encode `N`, `n_eval`, prior, success threshold,
posterior mode, and strict/bridge status.

## Apply to a disposable clone

The historical v3 reconstruction remains:

```bash
SOURCE=/data/robotixx/ued_bench/src/minimax-d053054
TARGET=/data/robotixx/ued_bench/src/minimax-frontier-v3-d053054
git clone --shared "$SOURCE" "$TARGET"

python3 ued_benchmark/scripts/apply_minimax_overlay.py \
  --target "$TARGET" --check
python3 ued_benchmark/scripts/apply_minimax_overlay.py \
  --target "$TARGET" --apply
python3 ued_benchmark/scripts/apply_minimax_overlay.py \
  --target "$TARGET" --check
```

The final check must report `pending_changes: 0`. The applicator refuses any
other Git HEAD and exact source anchors must match, making upstream drift fail
closed. It is idempotent and writes `.frontierrl_overlay.json` into the
disposable clone. `OVERLAY_CONTRACT.json` is content-addressed; its version and
SHA-256 are persisted in run metadata and validated by the runner.

For v4, use a different disposable clone and the versioned applicator:

```bash
TARGET_V4=/data/robotixx/ued_bench/src/minimax-frontier-v4-d053054
git clone --shared "$SOURCE" "$TARGET_V4"
python3 ued_benchmark/scripts/apply_minimax_overlay_v4.py \
  --target "$TARGET_V4" --check
python3 ued_benchmark/scripts/apply_minimax_overlay_v4.py \
  --target "$TARGET_V4" --apply
python3 ued_benchmark/scripts/apply_minimax_overlay_v4.py \
  --target "$TARGET_V4" --check
```

## Focused CPU verification

The source-faithful JAX environment prepared for this benchmark is:

```text
/data/robotixx/ued_bench/envs/minimax-jax0431-cpu
```

Run:

```bash
JAX_PLATFORM_NAME=cpu \
PYTHONPATH="$TARGET/src" \
/data/robotixx/ued_bench/envs/minimax-jax0431-cpu/bin/python \
  -m unittest -v ued_benchmark.tests.test_frontier_activity
```

The suite covers the formula, exact Beta expectation, Jensen ordering,
concrete-count validation, one-observation-per-stream semantics, posterior
accumulation, eviction reset, strict mismatch rejection, unchanged rank plus
staleness mixing, strict incomplete-group rejection, duplicate evidence, and
fail-closed checkpoint reload. Every PLR resume rejects missing or foreign
buffers and mismatches in all static fields stored by `PLRBuffer`; Frontier
resumes additionally preserve the posterior and overlay identity. The current
evidence lane deliberately rejects `n_devices>1` until the upstream sharded
buffer resume path is repaired.

The v4 adversarial suite additionally covers exact tie-block mass, exact
tie-mode probability permutation equivariance, all-equal and partially filled
buffers, raw distinct-score bit parity plus the frozen normalized float32 bound,
nonfinite fail-closed behavior, realized duplicate replay draws, parsed-config
matching, and both-direction checkpoint drift:

```bash
JAX_PLATFORMS=cpu PYTHONPATH="$TARGET_V4/src" \
/data/robotixx/ued_bench/envs/minimax-jax0431-cpu/bin/python -B \
  -m unittest -v ued_benchmark.tests.test_tie_aware_rank_v4 \
  ued_benchmark.tests.test_tie_aware_terminal_v4
```

The grouped LSTM layout also has a parsed-config integration smoke for both
the Frontier and matched MaxMC 4x8 arms. It verifies complete grouped
insertion on the first cycle and a forced-replay PPO update on the second:

```bash
JAX_PLATFORM_NAME=cpu \
PYTHONPATH="$TARGET/src" \
/data/robotixx/ued_bench/envs/minimax-jax0431-cpu/bin/python \
  -m unittest -v ued_benchmark.tests.test_grouped_runner_smoke
```

The exact grouped interpretation additionally requires RNG separation across
the eight copies of each task. A focused contract test verifies shared reset
keys within each level, pairwise-distinct step keys across all 32 streams, and
non-broadcast categorical action draws under the pinned stack:

```bash
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu \
PYTHONPATH="$TARGET/src" \
/data/robotixx/ued_bench/envs/minimax-jax0431-cpu/bin/python \
  -m unittest -v ued_benchmark.tests.test_grouped_rng_contract
```

Passing this test establishes the RNG layout, not statistical independence of
realized returns; shared policy parameters and deterministic dynamics remain
part of the task-conditional experiment.

Generate full commands with:

```bash
JAX_PLATFORM_NAME=cpu PYTHONPATH="$TARGET/src" \
/data/robotixx/ued_bench/envs/minimax-jax0431-cpu/bin/python \
  -m minimax.config.make_cmd \
  --dir "$PWD/ued_benchmark/configs" \
  --config maze_frontier_exact_grouped_n8
```

## Scientific scope

This overlay is a score-only intervention and an acquisition hypothesis. It
does not establish that coefficient activity is a gradient norm, learning
progress, or a universally optimal curriculum. The two posterior modes and the
bridge arm are development ablations. A benchmark-beating claim requires a
frozen protocol, matched interaction and update budgets, the full held-out
AMaze suite, and paired multi-seed uncertainty against robust PLR and ACCEL.
