# Frozen maze wave-2 AUC multiverse

**Status: `insufficient_raw_trajectories`.**

The analysis unit is an independent seed/warm-start block. The two
samplers are repeated observations inside each block and are never
counted as twelve independent replicates.

## Frozen summary anchor

| view | positive blocks | mean MaxRL - GRPO | range |
|---|---:|---:|---:|
| uniform | 6/6 | +0.01496 | [+0.00361, +0.03646] |
| frontier_un | 6/6 | +0.02404 | [+0.00641, +0.03405] |
| sampler average within block | 6/6 | +0.01950 | [+0.00661, +0.03025] |

## Why the requested robustness result cannot be computed

Only 0/24 required checkpoint JSONL files are present. The committed factorial JSON stores one AUC scalar plus
initial/final endpoints per cell; that is not enough to reconstruct
checkpoint order or contributions.

Therefore simple-mean versus trapezoid, warmup inclusion, early/mid/full
horizons, leave-one-checkpoint-out, minimum sign count, and effect range
are **not estimable**. Reporting them from the summaries would invent data.

Required next input: the 24 files
`fact250_{uniform,frontier_un}_{maxrl,grpo}_s{6..11}.jsonl` from
execution fork `9f7dd2e`. Re-running this script will first reproduce every
legacy `cov_auc_delta` exactly, then execute the frozen specification.

The machine-readable result enumerates every missing path and records the
source checksum, so this failure is deterministic and auditable.
