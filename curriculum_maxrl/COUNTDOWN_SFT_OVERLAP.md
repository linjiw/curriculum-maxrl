# Countdown SFT/evaluation overlap repair

## Current result

The original source audit (reported as F12 in the non-vendored experiment
repository and independently recorded in
`FINAL_REVIEW_RESPONSE_AND_GUIDANCE_2026-08-07.md`) found the following under
the paper's task identity `(target, sorted operand multiset)`:

| evaluation tier | tasks | SFT-exposed | clean |
|---|---:|---:|---:|
| tier 0 | 128 | 27 | 101 |
| tier 1 | 128 | 0 | 128 |
| tier 2 | 128 | 0 | 128 |

Consequently, absolute tier-0 values computed over all 128 tasks are marked
contaminated.  The headline tier-1 sharpening comparison and tier-2 results do
not have SFT/evaluation task overlap under this identity.

## Reproduction

`audit_countdown_sft_overlap.py` recomputes the overlap from JSON, JSONL, or
Parquet exports and records SHA-256 hashes for every input.  It also emits the
101 clean tier-0 task keys.  With structured per-task outcomes, it derives
clean mean@N and pass@N endpoints automatically:

```bash
python3 curriculum_maxrl/audit_countdown_sft_overlap.py \
  --sft /path/to/sft_examples.parquet \
  --eval /path/to/frozen_eval.parquet \
  --outcomes /path/to/per_task_step60_outcomes.jsonl \
  --output /tmp/countdown_sft_overlap.json
```

The current repository contains only tier-level Countdown endpoint aggregates.
It does **not** contain the SFT example manifest, the frozen evaluation task
manifest, per-task sampled outcomes, or compatible model checkpoints.  Thus,
the earlier 27/0/0 counts can be preserved and disclosed here, but cannot be
recomputed from this checkout alone.

The exact minimum prerequisite for the 101-task reanalysis is, for every
retained arm and seed at step 60:

1. the frozen evaluation task's target and operand multiset; and
2. all 16 binary verifier outcomes for that task.

Aggregate mean@16/VERL-bootstrap-best@16 values are not invertible to the
task-level subset, and the latter cannot be converted to standard pass@16.
If those outcome rows no longer exist, the alternative is checkpoint inference
on the frozen 101-task manifest with the original decoding configuration.
