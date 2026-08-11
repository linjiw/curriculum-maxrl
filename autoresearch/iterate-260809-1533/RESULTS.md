# Countdown recovery and GPU pilot results

Date: 2026-08-09

## Research decision and outcome

The handoff's highest-value next study was E2: a three-seed, genuinely
dose-matched live-group replay control for Countdown. The older
`ppo_epochs=2` replay is higher-dose and cannot support a dose-matched claim.
This iteration rebuilt the missing local execution prerequisites, implemented
the fixed-slot control, froze the protocol, and executed it on the GPU.

The result is a treatment-delivery finding rather than an endpoint comparison:

- prospective current-batch E2 stopped before seed 1 step 12 because no
  informative live source existed;
- the separately preregistered recent-buffer E2b passed a seed-1 pilot and full
  run, but stopped before seed 2's first replay optimizer update because its
  best exact source selection missed the 5% auxiliary-token gate; and
- consequently neither three-seed direction test is interpretable. No gate was
  loosened and no failed run was resumed.

## Recovered runtime

- GPU: NVIDIA GeForce RTX 5090, 32 GB; CUDA 12.8; PyTorch 2.7.0.
- Runtime: official `tajwarfahim/maxrl` at commit
  `7197bbb46a2ecd866da52f6b401ff20a34fe9390`, patched reproducibly under
  `/data/robotixx/curriculum-maxrl-runtime/maxrl`.
- Python overlay: `/data/robotixx/curriculum-maxrl-runtime/venv`.
- Frozen base model: `HuggingFaceTB/SmolLM2-360M-Instruct` revision
  `a10cc1512eabd3dde888204e902eca88bddb4951`.
- Clean SFT checkpoint:
  `/data/robotixx/curriculum-maxrl-runtime/models/countdown_sft_clean_v1`.

The single-GPU runtime uses Gloo for the singleton process group and bypasses
FSDP only when world size is one. On this host, singleton NCCL collectives hang
and Torch 2.7 `FSDP(NO_SHARD)` backward also hangs. Multi-GPU behavior remains
on the original NCCL/FSDP path.

## Leakage-free Countdown v2

Artifacts are in
`/data/robotixx/curriculum-maxrl-runtime/data/countdown_v2_rebuilt`.

- Train: 10,000 unique tasks; tier counts 2,000 / 4,000 / 4,000.
- Test: 384 unique tasks; 128 per 2/3/4-operand tier.
- SFT: 6,000 examples drawn only from the RL train split.
- Train/test overlap: 0; SFT/test overlap: 0.
- Test SHA-256:
  `95b1456fc3f49bc6f463614fef92900d748a07e4b429fd9383bbcf5edcb4e489`.
- SFT JSONL SHA-256:
  `c8c4e2bb067cf1a3c41f1748a50d22a77dc1bb4f0a642d4763df59082b0ad970`.

All 6,000 SFT expressions pass the exact `Fraction`/AST verifier and no SFT
row was truncated. Training took 188 optimizer steps (35.15 s), ended at loss
0.134957, and peaked at 8,029.98 MiB allocated GPU memory.

## Frozen pass@16 probes

Each row used all 128 test tasks in the tier, 16 samples per task,
temperature 1.0, top-p 1.0, 128 maximum new tokens, and seed 1234.

| checkpoint | tier (operands) | mean@16 | pass@16 | relabelable failures / failures |
|---|---:|---:|---:|---:|
| raw | 0 (2) | 0.0000 | 0.0000 | 0 / 2,048 (0.0000) |
| raw | 1 (3) | 0.0000 | 0.0000 | 0 / 2,048 (0.0000) |
| raw | 2 (4) | 0.0000 | 0.0000 | 0 / 2,048 (0.0000) |
| clean SFT | 0 (2) | 0.3950 | 0.9141 | 634 / 1,239 (0.5117) |
| clean SFT | 1 (3) | 0.0874 | 0.6406 | 895 / 1,869 (0.4789) |
| clean SFT | 2 (4) | 0.0068 | 0.0938 | 507 / 2,034 (0.2493) |

The preregistered landscape gate passes: tier 1 is a learnable band and tier 2
is a frontier. Tier 0 is saturated by the SFT checkpoint and should not be the
primary E2 endpoint.

## End-to-end training pilots

Baseline MaxRL completed one full generate/reward/log-prob/advantage/backward/
optimizer step in 12.557 s, with 6.755 GiB maximum allocation and score mean
0.0625.

The B2 hindsight integration pilot completed one step with 8 prompts x 16
rollouts. It observed 3 dead groups, accepted 3 relabeled groups (11 rollouts),
and moved success rollouts from 13 to 24. Exact dose telemetry was:

- accepted auxiliary group response tokens: 604, 610, 569 (1,783 total);
- optimizer rows: 128;
- optimizer response tokens: 4,544;
- step time: 39.434 s; maximum allocation: 6.755 GiB.

This is an integration and accounting check, not a comparative scientific
result.

## E2 current-batch treatment delivery

The frozen protocol is in `E2_PREREG.md`; the failure record is in
`E2_FIXED_SLOT_FAILURE.md`. B1 and B2 seed 1 completed 60/60 steps. The replay
arm completed 11 valid optimizer steps, matching 52 requested groups with no
fallback, 15.4% informative-slot displacement, 2.20% cumulative auxiliary-token
mismatch, and 1.52% cumulative total optimizer-token mismatch. Before step 12,
B2 requested replay but the current replay batch had no informative source.
The strict run stopped before updating.

## E2b recent-buffer follow-up

`E2B_PREREG.md` was frozen after the E2 failure and before any buffer-replay
outcomes. It retained every E2 setting and added a 64-group buffer with a
maximum source age of eight optimizer steps.

The 12-step seed-1 pilot passed the former failure point. The clean seed-1 full
run also passed all delivery gates: 60 audit rows, 329 scheduled groups, 289
buffer-source uses, 9 displaced informative slots (2.74%), no fallback, maximum
source age 8, 2.43% maximum cumulative auxiliary-token mismatch, and 0.97%
maximum cumulative optimizer-token mismatch. Final mismatches were 0.061% and
0.062%, respectively.

Seed-1 held-out endpoints are descriptive only because the three-seed test did
not complete:

| arm | tier-0 mean@16 | tier-0 pass@16 | tier-1 mean@16 | tier-1 pass@16 | tier-2 mean@16 | tier-2 pass@16 |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 0.6040 | 0.8516 | 0.1987 | 0.6172 | 0.0259 | 0.1953 |
| B2 | 0.5405 | 0.6406 | 0.1265 | 0.2344 | 0.0230 | 0.0547 |
| Rb | 0.4424 | 0.4453 | 0.0864 | 0.0938 | 0.0000 | 0.0000 |

Within this single seed, both auxiliary arms were worse than B1 and Rb was
worse than B2 on the primary tier-1 mean@16 and safety pass@16 endpoints. These
values do not establish a direction across seeds.

Seed 2 B1 and B2 then completed 60/60 steps with validated Hugging Face
checkpoints. Before seed 2 Rb's first optimizer update, strict delivery stopped:

- B2 requested 5 auxiliary groups totaling 3,031 response tokens;
- the matched generated batch contained only 2 informative source groups;
- the exact dynamic-programming matcher selected the best admissible five
  sources, totaling 2,850 tokens;
- auxiliary mismatch was 181 / 3,031 = 5.9716%, over the frozen 5% gate;
- total optimizer-token mismatch would have been 181 / 4,716 = 3.8380%; and
- scheduled slots were present, with no fallback or informative-slot
  displacement.

A separate one-step relaxed-ceiling diagnostic reproduced and logged this same
batch geometry. It made no endpoint checkpoint and is excluded from E2b. Seed
3 was not launched after the preregistered inconclusive condition was met.

## Research conclusion and next experiment

Recent buffering solves later empty-current-batch failures after the buffer has
been populated, but it does not guarantee cold-start token support. E2 and E2b
therefore do not answer whether exact hindsight relabeling beats a generic
matched auxiliary update; they show that the current placebo construction is
not reliably deliverable under the frozen constraints.

The next comparison should be a separately preregistered E2c. It should require
a delivery-only preflight for every seed before any endpoint run and eliminate
the empty cold start, preferably with an immutable, train-only source reservoir
generated from the same frozen SFT checkpoint. The preflight must demonstrate
all per-step group/slot gates and both 5% cumulative token gates across every
planned seed. Its source-generation budget and any loss-weight consequences
must be specified symmetrically before training. Do not reinterpret E2b by
loosening its 5% gate or seeding its buffer post hoc.
