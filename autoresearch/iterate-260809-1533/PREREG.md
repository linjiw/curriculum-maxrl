# Autoresearch iteration: local Countdown recovery

Frozen before the first full-dataset GPU probe on 2026-08-09.

## Scope

Recover the missing execution environment and inputs needed for the handoff's
next recommended experiment, E2 (a genuinely dose-matched live-group replay
control). The prior EC2 execution fork, custom Countdown pool builder, and SFT
checkpoint are not present in this checkout, so no old endpoint will be treated
as directly reproducible until those gaps are rebuilt and measured.

## Iteration 1: raw-model syntactic-prerequisite probe

- Model: `HuggingFaceTB/SmolLM2-360M-Instruct`, immutable HF revision
  `a10cc1512eabd3dde888204e902eca88bddb4951`.
- Dataset: all 384 held-out tasks in rebuilt Countdown v2
  (`test.parquet` SHA-256
  `95b1456fc3f49bc6f463614fef92900d748a07e4b429fd9383bbcf5edcb4e489`),
  128 tasks in each 2/3/4-operand tier, zero train/test and SFT/test overlap.
- Sampling: 16 independent samples per task, temperature 1.0, top-p 1.0,
  maximum 128 new tokens, seed 1234.
- Primary metric: fraction of failed samples that are exact-verifier-valid for
  some positive integer destination (`relabel_yield_on_failure`).
- Secondary metrics: mean@16 and pass@16 by tier.

Prediction: the raw checkpoint has relabel yield at or below 1% in every tier,
confirming the paper's claimed syntactic prerequisite for format SFT. If any
tier exceeds 1%, test a raw-model RL pilot before training a new SFT checkpoint.
Otherwise train the clean 6,000-example SFT split and re-run the identical
probe. Do not interpret RL arm contrasts until the post-SFT landscape has a
nonzero learnable band and at least one frontier tier.

The SFT configuration is frozen as: one epoch, maximum sequence length 384,
micro-batch 16, gradient accumulation 2 (effective batch 32), AdamW with
learning rate 5e-5, cosine decay, 10% warmup, weight decay 0.01, bf16, seed
2026, and completion-only loss. The landscape gate passes only if at least one
tier has pass@16 in [0.10, 0.80] and at least one harder tier has pass@16 <=
0.20. All-zero means insufficient SFT; all tiers above 0.80 means saturation.

## E2 gate

No full E2 arm may launch until:

1. the MaxRL/verl smoke run completes on this CUDA 12.8 / RTX 5090 stack;
2. data and SFT/test overlap are both zero;
3. auxiliary accepted-group, token, loss-weight, and optimizer-dose accounting
   are emitted automatically; and
4. the baseline, hindsight, and matched-replay arms share one frozen SFT
   checkpoint, fixed seeds, and a committed pass/fail/inconclusive branch.
