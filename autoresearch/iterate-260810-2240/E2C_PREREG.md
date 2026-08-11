# E2c preregistration: immutable-reservoir dose-matched replay

Frozen on 2026-08-10 before reservoir generation, E2c training, or any E2c
held-out evaluation. E2 and E2b remain separate, treatment-delivery-
inconclusive experiments.

## Pre-outcome metric provenance clarification (2026-08-10)

The historical Countdown artifacts called VERL's 1,000-resample,
with-replacement `best@16` statistic `pass@16`. That scalar is a bootstrap
coverage proxy, not standard unbiased pass@16, and its missing task outcomes
make retrospective conversion impossible. E2c does not reuse that evaluator.
`curriculum_maxrl.countdown.eval_countdown` retains all 16 binary verifier
outcomes for every held-out task; at $n=k=16$, its `pass@16` is the standard
observed-set indicator that at least one of the 16 samples succeeds. The locked
endpoint analyzer must recompute both `mean@16` and `pass@16` from those raw
outcomes and reject a summary/raw mismatch. This clarification changes no arm,
seed, gate, data, generation setting, endpoint, or decision branch and was
recorded before any E2c held-out evaluation.

The paired evaluator reseeds CPU and CUDA immediately after loading each model,
so checkpoint-load internals cannot consume an arm-specific prefix of the RNG
stream. Every summary records hashes and byte sizes for the exact config and
weight files it evaluated plus the frozen evaluator and verifier hashes. The
endpoint analyzer recomputes the model hashes and requires them to match the
last outcome-blind post-delivery readiness receipt.

During generation, per-arm stdout (including its summary) is sealed in an
arm-specific log. The driver does not expose a result until all nine arm/seed
summary and raw-outcome pairs exist and the complete matrix analyzer passes.

## Change from E2b

All E2 arm definitions, seeds, 60-step training settings, exact B2 accepted
replacement slots, 5% cumulative token gates, fixed-checkpoint rule, held-out
evaluation, endpoints, and decision branches remain unchanged except for the
generic replay source:

- E2c may draw only from one immutable source reservoir generated from the
  frozen clean-SFT checkpoint on train-split prompts.
- It may not draw from the current batch or a policy-evolving recent buffer.
- Reservoir sampling is with replacement, but a source whose dataset index
  equals the scheduled target slot is ineligible for that slot.
- Old log probabilities are recomputed under the current policy after the
  frozen group is inserted. E2c is therefore an explicitly off-policy
  weighted-replay placebo, not an unbiased policy-gradient estimator.

This change eliminates cold-start support failure and makes source-age
semantics constant and explicit. It does not assert that frozen-SFT replay is
identical to live replay; the comparison isolates exact hindsight direction
against a deliverable, generic informative auxiliary direction.

## Frozen reservoir protocol

- Source checkpoint:
  `/data/robotixx/curriculum-maxrl-runtime/models/countdown_sft_clean_v1`.
- Its recorded base-model revision is
  `a10cc1512eabd3dde888204e902eca88bddb4951`.
- Source data:
  `/data/robotixx/curriculum-maxrl-runtime/data/countdown_v2_rebuilt/train.parquet`.
- Generation seed: 424242.
- Generation budget: 60 batches x 8 prompt groups x 16 rollouts, maximum 128
  response tokens, temperature 1.0, top-p 1.0.
- Model update during collection: learning rate exactly 0; collection happens
  after exact reward computation and before the no-op optimizer update.
- Retention: the first 256 unique-by-dataset-index informative groups in
  deterministic dataloader order. An informative group has at least one
  success and at least one failure under its original requested target.
- Minimum reservoir gate: at least 128 retained groups, all of size 16, with at
  least 16 distinct aggregate response-token counts. Failure stops E2c before
  any training arm.
- Artifact: CPU tensors plus normalized non-tensor metadata, accompanied by a
  JSON manifest and SHA-256 checksum. The completed artifact is immutable.

No evaluation task, evaluation completion, hindsight-rewritten response, or
policy-updated rollout may enter the reservoir.

## Frozen input fingerprints and executable lock

The following SHA-256 fingerprints were recorded on 2026-08-10 before
reservoir generation, E2c training, or E2c held-out generation:

- `train.parquet`: `ac0671a2215a806d6c75b359f21697a9cfc8d8f47eef3b236391bd6e9fada91a`
- `test.parquet`: `95b1456fc3f49bc6f463614fef92900d748a07e4b429fd9383bbcf5edcb4e489`
- clean-SFT `config.json`: `283834b57c6e55af57e59b007df3bfcaf2f898dbb22fb535a46d224b73acb0cd`
- clean-SFT `model.safetensors`: `3198bb0f0c8598ec9aa713540e19472ebbe8363702db0d555fb060c679128ff8`
- clean-SFT `tokenizer.json`: `7d27c493c729a66ecefc837280b05d948b1ed50d130eebdbf911b1b36cf38ed7`
- clean-SFT `training_metrics.json`: `36085b432f5d3bed12e192648e996de6a10c41a60f9955582cce56b5bd8589f4`
- exact Countdown verifier/reward implementation: `99c04d4a4914170a528c67337aec364e7410074c552d9848c714f78c0f9e2312`
- paired raw-outcome evaluator: `0f642db64cabff66631b7e9ac88f1f3519651b21bee351051a1190f1a5bf653d`

The reward file's recorded filesystem modification/change time is 2026-08-09
15:41:21 America/New_York, before the first reusable comparator log began at
17:32:47. Every comparator log also names this exact path and the `compute_score`
entry point. This is supporting local provenance, not a substitute for the
content fingerprint enforced on all remaining launches.

The executable protocol rejects a mismatch in any fingerprint, the pinned
MaxRL commit, the ordered seed tuple `(1, 2, 3)`, 60-step budget, 5% gates,
group/row counts, or the 4,096 MiB launch ceiling. Scientific environment
overrides are ignored by the E2c driver. Operational path overrides remain
permitted only where the content fingerprints and runtime-code parity still
pass.

The 31-file research/patch/runtime manifest
`E2C_CODE_MANIFEST.json` has SHA-256
`0e46b89fcc01300b52bc6fc4e0c8a0ee5f2aa72d357b233e85c357972e8d3828`.
It also records the Python, PyTorch/CUDA, Transformers, FlashAttention, Ray,
Hydra, pandas, PyArrow, NumPy, and verl versions. The orchestrator verifies that
manifest before every readiness or launch path; the orchestrator itself is
15,676 bytes with SHA-256
`729447c426944f060b88cae272d537fe78a89e61bc0db3c1b6467daebc2cd4b9`.

## Preflight and execution order

1. Complete B2 training for all three fixed seeds without held-out endpoint
   evaluation; freeze each 60-row dose-accounting schedule.
2. Generate and freeze the shared reservoir.
3. Run the static preflight over all three schedules and the reservoir. It must
   validate provenance, train/test disjointness, group informativeness, exact
   step/slot metadata, non-self source support, and prospective auxiliary-token
   support under the 5% cumulative gate.
4. Train all three E2c replay arms and validate every runtime delivery audit.
5. Only if all three pass, evaluate all fixed B1/B2/E2c step-60 checkpoints.

No partial held-out endpoint is inspected between steps 1 and 4.

## Runtime validity gates

For every step and seed:

1. the B2 schedule has one unique row for that step and complete accepted-slot
   metadata;
2. E2c replaces exactly those dataset-index slots and `fallback_slots=0`;
3. replay group count equals B2 accepted relabel-group count;
4. every source is reservoir-backed, informative, train-only, and distinct
   from its target dataset index;
5. optimizer rows equal 128; optimizer-step and learning-rate schedules match;
6. cumulative auxiliary response-token mismatch is at most 5%; and
7. cumulative total optimizer response-token mismatch is at most 5%.

Any violation stops the affected seed before its optimizer update and makes
the three-seed direction test inconclusive. The source budget, reservoir,
matcher, or gates are not changed under the E2c label.

As frozen in E2, `displaced_live_slots` is also reported against a 25% share of
all scheduled slots. Exceeding 25% does not suppress already delivery-valid
fixed-slot endpoints; it forbids the pure extra-dose interpretation and limits
the contrast to fixed-slot direction substitution. This threshold is a
prospective interpretation branch, not a tunable runtime gate.

## Evaluation and analysis

The evaluation protocol and primary/safety endpoints are exactly those in
`../iterate-260809-1533/E2_PREREG.md`, with the replay arm renamed E2c. Results
are paired by training seed and described at the seed level; no independent-
rollout p-value is reported with n=3.

## Manuscript branch

- Valid three-seed result: use only the decision branch supported by both
  tier-1 endpoints and show all seed-level contrasts.
- Delivery failure: label E2c treatment-delivery inconclusive, keep the
  historical higher-dose replay as an upper-bound control, and remove any
  claim that the LLM-scale direction term was isolated.
