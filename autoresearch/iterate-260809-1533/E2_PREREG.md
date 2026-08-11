# E2 preregistration: dose-matched live-group replay

Frozen on 2026-08-09 after implementation/unit tests and a five-step treatment-
delivery pilot, but before any 60-step endpoint run on the rebuilt artifacts.
The pilot was used only to debug delivery and set feasibility gates; it had no
held-out evaluation and is excluded from endpoint analysis.

## Question

Does B2 Countdown hindsight improve tier-1 mean accuracy because it supplies
more nonzero optimization dose, or because the exact relabeled direction is
valuable? The historical `ppo_epochs=2` arm is a higher-dose upper bound, not a
dose-matched control.

Because the historical execution fork, exact pool, and checkpoint are absent,
B1 and B2 will be re-estimated alongside replay. Historical endpoints will not
be mixed with this rebuilt experiment.

## Frozen artifacts

- Runtime: `tajwarfahim/maxrl` commit
  `7197bbb46a2ecd866da52f6b401ff20a34fe9390` plus the reproducible patches in
  `verl_integration/`.
- Initial model: `/data/robotixx/curriculum-maxrl-runtime/models/countdown_sft_clean_v1`.
- Train data: `/data/robotixx/curriculum-maxrl-runtime/data/countdown_v2_rebuilt/train.parquet`.
- Held-out data: `/data/robotixx/curriculum-maxrl-runtime/data/countdown_v2_rebuilt/test.parquet`,
  SHA-256 `95b1456fc3f49bc6f463614fef92900d748a07e4b429fd9383bbcf5edcb4e489`.
- Training seeds: 1, 2, 3. Evaluation seed for training seed `s`: `10000+s`,
  shared across arms.

## Arms and fixed training protocol

- B1: MaxRL, no hindsight, no replay.
- B2: MaxRL plus exact Countdown hindsight on eligible all-fail groups.
- R: MaxRL plus live-group replay into the exact dataset-index slots B2
  accepted at each step for the same seed.

For every arm: 60 optimizer steps; 8 generated prompt groups per step; 16
rollouts per group; maximum response length 128; temperature 1.0; top-p 1.0;
one PPO epoch and one optimizer step per generated batch; batch/mini-batch 8;
micro-batch 4; AdamW learning rate 1e-5 with the same upstream schedule; MaxRL
advantages; hindsight/replay coefficient 1.0; no KL loss; one frozen step-60
checkpoint. B2 must finish before R for each seed because its immutable JSONL
audit is R's schedule.

R samples informative current live groups with replacement. It writes them
into B2's exact accepted prompt slots while preserving each target uid, so the
optimizer still sees 128 rows and eight 16-way MaxRL groups. A source may not
replay itself. Sources are selected by deterministic dynamic programming to
jointly minimize cumulative auxiliary-response-token and total optimizer-
response-token mismatch.

After policies diverge, a B2-accepted slot can be informative in R. Fixed-size
on-policy training cannot both keep that direction and insert the scheduled
replay group. R therefore replaces the slot and audits it as
`displaced_live_slots`; this makes R a fixed-slot direction-substitution
control. If more than 25% of all scheduled slots are informative when replaced,
the pure extra-dose interpretation is declared inconclusive and only the
fixed-slot contrast is reported. This threshold was frozen after the five-step
feasibility pilot (2/21 displaced slots) and before full runs.

## Treatment-delivery gates

An R seed is valid only if all of the following hold at every step:

1. generated group count, optimizer steps, learning-rate schedule, and final
   checkpoint step match its B2 source;
2. replay group count exactly equals B2 accepted relabel-group count;
3. optimizer rows equal 128 and replay uses B2's exact accepted dataset slots
   (`fallback_slots=0`);
4. cumulative auxiliary response-token mismatch is at most 5%;
5. cumulative total optimizer response-token mismatch is at most 5%; and
6. an informative non-self source exists for every scheduled slot.

Failure of any gate stops that seed and makes the three-seed direction test
inconclusive. The strict runtime raises immediately for gates 2--6. B2 audits
must contain 60 unique steps and accepted-slot metadata. Actual wall time and
the displaced-slot fraction are reported, not optimized post hoc.

## Frozen evaluation and endpoints

Each final Hugging Face checkpoint is sampled on all 384 held-out tasks (128
per tier) with 16 draws per task, temperature 1.0, top-p 1.0, maximum 128 new
tokens, and the seed pairing above. No intermediate checkpoint is selected.

- Primary: within-seed tier-1 `mean@16` contrast B2-B1 and R-B1, summarized by
  the mean of the three seed-level paired contrasts.
- Safety: the analogous empirical tier-1 `pass@16` contrasts, where a task
  passes if any of its 16 samples is correct.
- Direction test: within-seed tier-1 R-B2 contrasts on both metrics.
- Tier 0 is secondary because clean SFT pass@16 is already 0.9141. Tier 2 is
  exploratory because clean SFT pass@16 is 0.0938.

With only three training seeds, results are descriptive paired estimates; no
asymptotic p-value or independent-rollout significance claim will be made.

## Decision branches

- If R reproduces B2's mean gain without B2's coverage loss, the gain is not
  direction-specific on this pool.
- If B2 beats R at matched delivered dose, the relabeled direction contributes
  measurable value.
- If B2 and R both lose coverage versus B1, the damage is a generic auxiliary-
  dose effect rather than hindsight-specific.
- If a delivery gate or the displaced-slot diagnostic fails, the direction
  claim is inconclusive; report the failure without substituting a new arm.

