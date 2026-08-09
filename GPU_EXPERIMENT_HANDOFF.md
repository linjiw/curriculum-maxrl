# GPU experiment handoff: the two studies worth running next

Status: design handoff, 2026-08-09. No GPU job was run while preparing this
document. This is a prospective plan, not a registration or a source lock.
The protocol, implementation, runtime, data, seed schedule, and analysis must
be independently reviewed and sealed before any scientific run begins.

## Executive decision

Run exactly these two GPU lanes, in this order:

| rank | lane | decision | why it is worth the compute | hard condition before launch |
|---:|---|---|---|---|
| 1 | **A. Estimator-specific learning-rate calibration and maze confirmation** | **GO after instrumentation/analyzer implementation and an engineering pilot** | This directly addresses the cleanest remaining reviewer objection: MaxRL and GRPO were compared at one common learning rate even though their coefficient scales differ. It also tests the paper's distinctive estimator-by-sampler prediction on the existing 1.26M-parameter model. | Instrument and lock the runner/analyzer, retain full evaluation outcomes, and demonstrate peak reserved GPU memory at or below 9.0 GiB. |
| 2 | **B. Truly dose-matched Countdown relabel versus live replay** | **CONDITIONAL GO; do not launch from this checkout yet** | This is the most valuable causal repair for the applied RLVR result. It can distinguish a relabel-direction effect from merely adding more gradient-bearing data. | Recover or rebuild and lock the missing Countdown v2 pool, SFT manifest/checkpoint, and execution fork; implement the paired-dose controller; then pass a 10-GB feasibility pilot. |

If only one lane can be afforded, run Lane A. It is much cheaper, its codebase
is local, and a positive registered interaction would materially strengthen the
main estimator/curriculum thesis. Lane B is more expensive and should not be
approximated with the old `ppo_epochs=2` arm: that arm updates every live group
twice while recycling affects at most 12 of 64 requested groups.

Neither lane licenses a claim of universal curriculum optimality, superiority
to full ProCuRL/PLR/SFL/PAIRED/ACCEL, or causal mediation by coefficient mass.

## What was checked locally

The following paths exist:

- `curriculum_maxrl/maze_gpu/train.py`, `model.py`, and `maze_env.py`;
- `curriculum_maxrl/maze_gpu_factorial/fact_analyze.py` and the two historical
  factorial launchers;
- `verl_integration/hindsight.py`, its immutable execution snapshot in
  `verl_integration/vendored/hindsight.py`, and the integration patches;
- `curriculum_maxrl/countdown_reviewer_arms/run_reviewer_arms.sh`; and
- `curriculum_maxrl/audit_countdown_sft_overlap.py`.

Three launch-readiness findings were reproduced:

1. During preparation, `train.py --help` initially failed on an undefined
   `AdvMassTeacher`. That import-time blocker has now been repaired in the
   shared worktree by aliasing the exact-`u_N` implementation, and
   `python3 curriculum_maxrl/maze_gpu/train.py --help` exits successfully.
   This validates the existing CLI surface only; it does not supply the new
   ledger, completed-update, lock, gate, or analysis contracts below.
2. The historical maze analyzer is not a safe analyzer for this study. It has
   no `p(1-p)` factorial, no learning-rate selection protocol, no strict raw
   schema, and its code includes the post-first-update `step=0` record while
   its prose says the AUC begins at step 25. A new completed-update coordinate
   and locked analyzer are required.
3. The Countdown wrapper refers to `smollm/countdown_a10g.sh`, but that file,
   the MaxRL execution checkout, the exact v2 task/SFT manifests, and the
   task-level outcomes are absent here. `verl_integration/smollm_a10g.sh` is a
   GSM8K launcher and is **not** a substitute.

The shell syntax of the historical launchers is valid. That is not evidence
that their external paths, environment, or data are present.

## Primary-source anchors

- The [MaxRL paper](https://arxiv.org/abs/2602.02710) defines a
  compute-indexed family of sampling objectives, and the audited official
  source is pinned at
  [`7197bbb46a2ecd866da52f6b401ff20a34fe9390`](https://github.com/tajwarfahim/maxrl/commit/7197bbb46a2ecd866da52f6b401ff20a34fe9390).
  Its released maze launcher uses one common learning rate and does not provide
  a multi-seed, estimator-specific LR protocol
  ([source](https://github.com/tajwarfahim/maxrl/blob/7197bbb46a2ecd866da52f6b401ff20a34fe9390/maze/maze_17.sh)).
- The official MaxRL implementation's centered-by-mean path and trainer
  routing are the source precedents for the practical estimator used here
  ([advantages](https://github.com/tajwarfahim/maxrl/blob/7197bbb46a2ecd866da52f6b401ff20a34fe9390/verl/trainer/ppo/core_algos.py#L402-L441),
  [routing](https://github.com/tajwarfahim/maxrl/blob/7197bbb46a2ecd866da52f6b401ff20a34fe9390/verl/trainer/ppo/ray_trainer.py#L270-L287)).
- Standard pass@k is the without-replacement estimator introduced in the
  [Codex evaluation paper](https://arxiv.org/abs/2107.03374). The official
  MaxRL LLM logger instead uses a with-replacement bootstrap proxy
  ([metric source](https://github.com/tajwarfahim/maxrl/blob/7197bbb46a2ecd866da52f6b401ff20a34fe9390/verl/trainer/ppo/metric_utils.py#L258-L296));
  the new Countdown run must therefore retain the underlying binary outcomes.
- [Hindsight Experience Replay](https://papers.nips.cc/paper_files/paper/2017/hash/453fadbd8a1a3af50a9df4df899537b5-Abstract.html)
  establishes achieved-goal relabeling as a sparse-reward data-reuse method.
  Our Countdown operation is narrower: a verifier-valid, per-row weighted-SFT
  auxiliary term inside a PPO-family loop, not an unbiased on-policy gradient
  and not the original off-policy HER algorithm.
- Official VERL exposes `trainer.validation_data_dir` and writes generation
  JSONL, including inputs, outputs, ground truth, and scores
  ([trainer source](https://github.com/verl-project/verl/blob/main/verl/trainer/ppo/ray_trainer.py#L2913-L2977),
  [validation dump](https://github.com/verl-project/verl/blob/main/verl/trainer/ppo/ray_trainer.py#L3286-L3306)).
  The exact pinned MaxRL fork must be tested rather than assumed to match
  current VERL.
- The official [SmolLM2-360M-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)
  identifies the 360M model and Apache-2.0 license. It does not establish that
  full-policy RL training fits a 10-GB GPU.
- PyTorch warns that reproducibility is not guaranteed across releases or
  platforms and documents deterministic controls
  ([reproducibility note](https://docs.pytorch.org/docs/stable/notes/randomness)).
  It also provides peak allocated/reserved memory counters
  ([CUDA memory semantics](https://docs.pytorch.org/docs/main/notes/cuda.html#memory-management)).
- ICLR 2027 is double blind, encourages reproducibility material, and permits
  anonymous code supplements
  ([author guide](https://iclr.cc/Conferences/2027/AuthorGuidelines)).

## Shared rules for both lanes

1. **Separate engineering, development, and confirmation.** Engineering
   seeds are never analyzed scientifically. Development may select a rate or
   validate dose feasibility. Confirmation uses fresh, never-before-run seeds.
2. **The paired seed block is the inferential unit.** Samplers, estimators,
   tiers, tasks, checkpoints, and evaluation samples within a block are not
   independent replicates.
3. **Lock before confirmation.** Hash the protocol, source plus transitive
   local dependencies, runtime lock, data/SFT manifests, command templates,
   closed JSON schemas, tests, and analyzer. Make the pre-execution commit or
   anonymous registration immutable. Do not call an internal timestamp alone
   a preregistration.
4. **Separate the outcome-blind gate from analysis.** The gate may inspect
   schemas, counts, hashes, finite values, task diversity, outcome variation,
   memory, and replay equality. It must emit no arm means, directions,
   intervals, or p-values. Only after every gate passes may the locked analyzer
   reveal contrasts.
5. **No outcome-dependent stopping.** Never inspect interim arm performance.
   Stop only for a prespecified mechanical failure: OOM, nonfinite tensors,
   corrupt/missing ledger, source mismatch, disk floor, or wall-clock safety
   limit in an engineering pilot.
6. **Evaluation is not training compute.** Report training generations,
   evaluation generations, gradient-bearing group exposures, optimizer steps,
   tokens, GPU-hours, and peak memory separately.
7. **One process per GPU.** The historical maze log reports OOM with two
   concurrent generation jobs. Sequential execution also makes wall-clock and
   failure accounting cleaner.
8. **Standard pass@k comes from raw task outcomes.** For task `i`, `n` sampled
   outputs, and `c_i` successes,

   `pass@k_i = 1 - C(n-c_i,k) / C(n,k)`.

   Average this per task. Do not convert the existing VERL bootstrap
   `best@k` proxy.
9. **Normalized AUC is fixed before outcomes.** For common evaluation
   coordinates `0=x_0<...<x_J=B` and metric `m_j`, use

   `AUC(m) = (1/B) * sum_j (x_j-x_{j-1})*(m_j+m_{j-1})/2`.

   Report `AUC(m)-m_0` as an interpretable change-from-warm-start diagnostic,
   but arm contrasts use `AUC(m)`; the common baseline cancels within a block.
   Duplicate or missing coordinates are fatal, not silently interpolated.

---

# Lane A — estimator-specific LR calibration and maze confirmation

## A1. Scientific question and allowed claim

Let `Y(e,s,b)` be target-uniform pass@8 AUC in confirmation block `b`, with
estimator `e` in `{practical_maxrl, grpo}` and sampler `s` in
`{uniform, p1mp, u32}`. Here `p1mp` is `p(1-p)` and `u32` is
`1-(1-p)^32-p`, both applied to the same decayed Beta teacher with the same
uniform floor.

The single registered primary contrast is

`I_b = [Y(maxrl,u32,b)-Y(maxrl,p1mp,b)]
       -[Y(grpo,u32,b)-Y(grpo,p1mp,b)]`.

Hypothesis A-H1: the mean of `I_b` is positive. The claim is supported only
if the paired mean is at least the frozen SESOI `+0.015`, the exact two-sided
paired sign-flip p-value is at most `.05`, and the 95% paired-block bootstrap
interval excludes zero.

Allowed wording after support: “On the frozen 17x17 maze design, the relative
advantage of the deployed-N score over the N=2 score was larger under
practical MaxRL than under GRPO after estimator-specific LR calibration.”
This is an estimator-by-sampler interaction, not proof that `u_N` is a
generally optimal curriculum.

Uniform comparisons are secondary. Common-LR results are a prespecified
sensitivity analysis and cannot rescue a failed tuned-LR primary.

## A2. Frozen treatment matrix

### Engineering only

- Seed `49000`.
- Two 25-update runs: `{practical_maxrl, grpo} x uniform` at LR `3e-4`.
- Same 1.26M model, N=32, eight groups/update, and the full evaluation batch.
- Purpose: memory/runtime/schema/determinism only. Never enter a plot, table,
  power calculation, or run registry of scientific evidence.

### Development: estimator-specific rate selection

- Seeds `50000,50001,50002`, disjoint from all historical seeds `0..11`.
- Sampler: **uniform only**, so sampler outcomes cannot influence LR choice.
- Matrix: `{practical_maxrl,grpo} x {3e-5,1e-4,3e-4} x 3 seeds` = 18 runs.
- Each estimator selects the LR with the largest three-seed mean target-uniform
  pass@8 AUC. If candidates are within `.002` AUC, select the one closest to
  `1e-4`; if still tied, select the smaller LR.
- Prespecified one-time boundary expansion: if the unique winner is `3e-5`,
  add `1e-5`; if it is `3e-4`, add `1e-3`, using the same three development
  seeds. No second expansion. A rate with any nonfinite/overflow run is
  ineligible. If no eligible rate remains for either estimator, Lane A is a
  no-go.

The chosen LR is estimator-specific but sampler-invariant.

### Confirmation

- Fresh paired seeds `51000..51011` (12 independent blocks).
- Tuned primary grid: `{practical_maxrl,grpo} x {uniform,p1mp,u32}` = 72 runs.
- Common-rate sensitivity: the same six cells at LR `1e-4`. If a tuned cell is
  exactly LR `1e-4`, bind the same raw artifact by hash rather than rerunning
  it. Thus confirmation is 72 to 144 unique runs.
- Randomize the arm order within each block from a sealed order file generated
  from a separate scheduling RNG. Every arm in a block uses the same exact
  warm-start checkpoint and held-out task manifest.

## A3. Held-fixed learner and compute

| item | frozen value |
|---|---|
| model | local `TinyTransformer`, `d_model=128`, six layers, about 1.26M parameters |
| environment | fresh 17x17 Prim mazes; 13 BFS-distance levels |
| SFT | 600 steps, geometric level mixture, one shared checkpoint per seed block |
| RL estimators | `weights_maxrl` (K=0 dropped) or sample-SD GRPO |
| rollouts | N=32 per task group |
| groups/update | 8 |
| budget | exactly 250 completed optimizer opportunities; complete every batch |
| optimizer | AdamW; all settings except selected LR identical |
| teachers | local `uniform`, `learnability`, and `frontier_un`; nonuniform floor `.15`, decay `.7`, power `1` |
| evaluation | completed updates `0,25,...,250`; fixed 16 tasks/level; 8 outcomes/task; evaluation RNG cannot mutate training/teacher RNG |
| primary metric | uniform over levels of standard pass@8, then normalized trapezoidal AUC |
| hindsight | off |

Record attempted and applied optimizer updates separately; a batch with no
nonzero weight is still a completed training opportunity and remains in the
budget. Training is matched by groups and rollouts, not wall-clock.

## A4. Power and sample-size rationale

The 12 surviving historical maze blocks provide a variance proxy, not pilot
evidence for the new `u32-p1mp` contrast. Recomputing the old
`(u32-uniform) x (MaxRL-GRPO)` AUC interaction gives block SD
`0.0176244`. A two-sided paired-t planning approximation at alpha `.05` gives
an 80%-power MDE of about `.01566` with 12 blocks (`.01321` with 16). The
registered analysis will instead use exact sign flips plus a paired bootstrap.

Twelve blocks are the recommended cost/precision compromise. If the team can
afford 24 additional tuned runs, increase to 16 blocks **before sealing** and
reserve seeds `51012..51015`; never add them after seeing 12-block results.

## A5. Raw artifact contract

Use closed schemas and reject duplicate JSON keys, nonfinite numbers, unknown
fields, scalar-to-vector broadcasting, missing records, and reordered event
coordinates. One run artifact must contain:

```text
schema: "maze_estimator_lr_v1"
artifact_state: "partial" | "complete" | "failed"
provenance:
  protocol_sha256, source_lock_sha256, runtime_lock_sha256,
  command_sha256, git_commit, dirty_diff_sha256_or_null
config:
  phase, block_seed, estimator, sampler, learning_rate,
  steps, groups_per_step, rollouts, sft_steps, eval_every,
  model_shape, teacher_decay, teacher_floor, teacher_power
manifests:
  warmstart_sha256, eval_task_manifest_sha256, arm_order_sha256
rng:
  named domain seeds and initial/final state hashes for actor init,
  SFT data, train task generation, train actions, teacher, evaluation
training_events[250]:
  completed_update, selected_levels[8], task_sha256[8],
  teacher_probability[8], p_hat_before[13],
  reward_bits[8][32], coefficient_weights[8][32],
  gradient_group_mask[8], loss, grad_norm, optimizer_applied,
  actor_before_sha256, actor_after_sha256, elapsed, memory
evaluation_events[11]:
  completed_updates, actor_sha256,
  tasks[13][16] {level, task_sha256, outcome_bits[8], return_or_length[8]},
  generation_count, evaluation_rng_before_sha256,
  evaluation_rng_after_sha256
accounting:
  groups=2000, train_rollouts=64000, eval_rollouts,
  attempted_updates=250, applied_updates, wall_seconds,
  peak_allocated_bytes, peak_reserved_bytes
failure: null | {stage, exception_type, message_digest, last_complete_event}
```

The stored summaries must be recomputed from these ledgers and exact-equal to
the raw-derived values. Do not store only per-level counts: preserve all eight
binary outcomes for every fixed evaluation task.

## A6. Outcome-blind gates

All must be true before analysis:

- exact source/runtime/protocol/data hashes and clean source lock;
- expected phase/seed/arm matrix, with no seed overlap;
- one byte-identical warm start and evaluation manifest across every arm in a
  block;
- exactly 250 training events and evaluation coordinates `0,25,...,250`;
- every training event has 8 groups x 32 rewards, all binary, and counters
  recompute to 2,000 groups and 64,000 training rollouts;
- all stored weights exactly match the locked estimator implementation;
- sampler probabilities are finite, nonnegative, sum to one, and recompute
  from the preceding teacher state; uniform is exactly uniform; each adaptive
  sampler becomes nonuniform at least once in pooled development;
- evaluation leaves every training/teacher RNG fingerprint unchanged;
- actor fingerprints change iff an optimizer update is applied;
- no NaN/Inf, no missing/extra event, no silent retry, and peak reserved
  memory at or below 9.0 GiB in the engineering pilot;
- pooled development has dead, mixed, and all-pass groups and nonconstant
  evaluation outcomes; and
- confirmation gate output contains no metric means, arm directions, CIs, or
  p-values.

## A7. Registered analysis

- Primary: `I_b` above on tuned-LR pass@8 AUC.
- Test: exact two-sided paired sign-flip over all `2^12` sign assignments.
- Interval: 100,000 paired-block bootstrap resamples, fixed analysis RNG seed
  `61000`, linear `.025/.975` quantiles.
- Decision: mean `>=.015`, p `<=.05`, and interval excludes zero.
- Secondary Holm family, alpha `.05`, in this fixed order:
  1. `u32-uniform` under MaxRL;
  2. `p1mp-uniform` under MaxRL;
  3. `u32-uniform` under GRPO;
  4. `p1mp-uniform` under GRPO;
  5. MaxRL-GRPO averaged over the three samplers.
- Report standard pass@k AUC for `k={1,2,4,8}`, endpoint pass@k, mean success,
  coefficient mass/group and per training rollout, dead/mixed/all-pass rates,
  task allocation, wall-clock, and evaluation/training generation counts.
- Common-LR interaction and all per-level/band decompositions are sensitivity
  analyses. Easy-band localization is descriptive unless separately powered.

## A8. Command templates and implementation TODOs

The core command below uses only arguments that exist in the local runner and
its `--help` surface now loads successfully. Scientific execution remains
blocked until the ledger, completed-update, memory, source-lock, and
strict-analysis TODOs above are implemented:

```bash
cd /absolute/path/to/curriculum-maxrl

# CLI exists; do not launch scientifically until the protocol additions land.
python3 curriculum_maxrl/maze_gpu/train.py \
  --teacher frontier_un \
  --estimator maxrl \
  --steps 250 \
  --seed 51000 \
  --tasks-per-step 8 \
  --rollouts 32 \
  --lr REPLACE_WITH_SEALED_MAXRL_LR \
  --sft-steps 600 \
  --eval-every 25 \
  --sft-ckpt gpu_lane_a_warmstart.pt \
  --out /absolute/external/run_root/seed_51000/maxrl_u32.jsonl
```

Teacher mapping in the existing CLI is exact:

- `uniform` -> uniform;
- `learnability` -> `p(1-p)`; and
- `frontier_un` -> `u_32` when `--rollouts 32`.

Do not use the historical `run_factorial*.sh` for this study: their seeds,
four-arm matrix, common LR, artifact names, and analyzer are historical.

Required TODOs before launch:

1. retain the repaired exact-`u_N` alias and add a `--help`/teacher-map
   regression test;
2. add full training/evaluation ledgers, memory counters, source/runtime lock
   binding, atomic incremental writes, and refuse-overwrite behavior;
3. replace ambiguous zero-based `step` with `completed_updates`;
4. add a strict gate/LR selector/analyzer and adversarial schema tests; and
5. run an independent pre-seal review, then create the immutable lock.

## A9. Expected cost and storage

The paper records approximately `0.7 A10G-hour` per 250-update maze run.
Therefore:

- development: 18 runs = about 12.6 GPU-hours; one boundary expansion adds
  4.2 hours;
- tuned confirmation: 72 runs = about 50.4 GPU-hours;
- common-LR sensitivity: 0 to 72 additional unique runs = 0 to 50.4 hours;
- expected total after engineering: roughly 63 to 118 A10G-hours.

The model and current aggregate JSONLs are small, but the proposed full ledger
is not. Budget 10-30 MB/run plus checkpoints, or under roughly 5 GB of compact
raw for the full worst-case matrix. Measure actual bytes in engineering and
reserve 3x the observed projection. Keep checkpoints outside Git.

---

# Lane B — genuinely dose-matched Countdown relabel versus live replay

## B1. Scientific question and allowed claim

The historical three-seed aggregate suggests that ungated relabeling changes
tier-1 mean@16 and the VERL bootstrap coverage proxy, but the existing replay
arm doubles optimizer epochs on every live group. Lane B asks a narrower
question:

> At the same base generations, batch tensor shape, optimizer epoch count,
> and number of formerly-zero groups converted into gradient-bearing groups,
> does verifier-valid achieved-goal relabeling differ from replaying ordinary
> live groups?

Let `M_b` be tier-1 mean@16 AUC in paired block `b`. The primary contrast is
`D_b = M_b(relabel)-M_b(live_replay)`.

Hypothesis B-H1 is supported only if mean `D_b >= +.020`, exact paired
sign-flip `p<=.05`, and the paired bootstrap 95% interval excludes zero.

Allowed wording after support: “On the frozen Countdown pool, verifier-valid
relabeling improved mean success beyond a live-replay control matched on
gradient-bearing group substitutions.” Always add that semantic target,
gradient norm, and task difficulty are not matched, and that the per-row
operation is weighted SFT rather than an unbiased on-policy gradient.

## B2. What “dose matched” means here

Both arms begin every step with the same 64 requested task groups and N=16
rollouts/group, from the same block-specific model checkpoint, task order, and
coordinate-based generation RNG domains. Both use one PPO epoch and keep a
64-group tensor.

The relabel arm runs the release-safe exact-verifier relabeler after original
reward computation and before old-log-prob computation. It is ungated and may
convert at most 12 all-fail groups. Let `d_t` be the number of groups actually
converted at step `t` (not merely attempted and not the number of relabeled
rows).

The paired live-replay arm consumes the relabel arm's sealed, hashed schedule
`(d_1,...,d_60)`. At step `t`, it:

1. identifies its own all-fail and naturally live groups from the original
   rewards;
2. selects `d_t` all-fail destination slots uniformly without replacement
   using a dedicated replay-slot RNG;
3. selects `d_t` naturally live source groups uniformly without replacement
   using a separate replay-source RNG;
4. replaces each selected all-fail group with one exact duplicate of a live
   source group, assigns a new group UID, and computes old log-probs under the
   unchanged prompt; and
5. performs the same single optimizer epoch on the same 64 x 16 tensor shape.

This converts exactly `d_t` otherwise-zero group slots into ordinary live
gradient groups, just as relabeling converts exactly `d_t` zero group slots
into relabeled gradient groups. It does **not** append a batch, double every
live group, rank replay sources by reward, or match gradient norm. Fixed padded
response tensors and `use_dynamic_bsz=false` are required; actual valid-token
counts remain a reported diagnostic.

The relabel run must execute first inside each paired block only to generate
the dose schedule. The runner may expose `d_t` and mechanical ledgers, but no
validation outcome or contrast, before the replay run completes. If replay has
fewer than `d_t` dead slots or live sources at any step, that development block
fails the feasibility gate. Do not cycle sources or silently reduce dose.

## B3. Frozen treatment matrix

### Engineering only

- Seed `51999`, one five-step paired job.
- Two arms: `relabel` and `paired_live_replay`.
- Try the faithful full-policy configuration first. Peak reserved memory must
  be `<=9.0 GiB` with at least 0.5 GiB device headroom.
- Prespecified memory fallback order, applied identically to both arms:
  1. PPO/log-prob microbatch 8 -> 4;
  2. 4 -> 2;
  3. enable CPU optimizer offload, then parameter offload, only if those exact
     keys exist in the pinned fork and a one-step numerical parity test passes.
- Quantization, LoRA, shorter responses, smaller N, fewer groups, and changed
  model are **not** memory fallbacks; they define a new experiment.

If no faithful configuration fits 10 GB, Lane B is a no-go on that device and
should move to a 16-24 GB rental GPU.

### Development

- Paired seeds `52000,52001,52002` = 6 runs.
- Full 60-step schedule, exact v2 pool/SFT, evaluation, and ledgers.
- Purpose: validate the dose controller, estimate paired variance without
  inspecting direction, and project memory/runtime/storage.
- Proceed only if all three pairs have exact dose equality, no replay shortage,
  native task/outcome variation, and paired primary SD `<=.030`. If SD exceeds
  `.030`, recalculate the required confirmation size before sealing; do not
  run an underpowered fixed 16-block confirmation by inertia.

### Confirmation

- Fresh paired seeds `53000..53015` (16 independent blocks) = 32 runs.
- Exactly two arms: release-safe ungated relabel and paired live replay.
- No baseline, gate, curriculum, one-target ablation, or `ppo_epochs=2` arm is
  added to this family after outcomes are known.

## B4. Held-fixed learner and evaluation

| item | frozen value |
|---|---|
| model | exact hashed SmolLM2-360M Countdown SFT checkpoint |
| task pool | exact hashed Countdown v2 train/eval manifests: permutation, parenthesization, exact division |
| estimator | practical MaxRL, N=16 |
| groups/step | 64 |
| budget | 60 completed updates, one optimizer epoch |
| LR | AdamW `1e-5` |
| response | max 1024 tokens; fixed tensor shape |
| curriculum/gate | both off |
| relabel | release-safe `verl_integration/hindsight.py`, per-row mode, at most 12 groups/step |
| evaluation | before training and completed updates `15,30,45,60`; 128 fixed tasks/tier x 16 outputs/task |
| primary population | clean tier 1; tier 2 secondary; tier 0 only after manifest-based 101-task decontamination |

Hash and report the model revision, tokenizer, chat template, SFT examples,
train/eval task identity, decoding temperature/top-p/top-k, and verifier. The
known task identity is `(target, sorted operand multiset)`. Tier 0 previously
had 27/128 SFT overlaps; tier 1 and tier 2 had none, but those counts must be
recomputed from the new manifests rather than copied as evidence.

## B5. Outcomes, AUC, and power

For every task and checkpoint retain all 16 binary verifier outcomes. Compute:

- `mean@16 = sum_i c_i / (128*16)`;
- standard pass@k for `k={1,2,4,8,16}` from the combinatorial formula;
- normalized trapezoidal AUC over updates `0,15,30,45,60` for each metric; and
- step-60 endpoints as prespecified secondary reads.

Primary: tier-1 mean@16 AUC relabel minus replay. Exact two-sided paired sign
flip over `2^16` assignments; 100,000 paired-block bootstrap resamples with
analysis RNG seed `63000`; SESOI `+.020`.

Holm secondary family, alpha `.05`, fixed order:

1. tier-1 standard pass@16 AUC;
2. tier-1 pass@8 AUC;
3. tier-1 mean@16 endpoint;
4. tier-1 pass@16 endpoint; and
5. tier-2 mean@16 AUC.

Report pass@2/pass@4, clean tier-0 outcomes, entropy, output length, exact
relabel/replay doses, relabeled rows, valid tokens, gradient norms, and task
identities descriptively. Do not label the legacy bootstrap proxy standard
pass@k.

For prospective scale, the historical B2 and higher-dose replay summaries
have across-seed sample SDs about `.014` and `.021` on tier-1 mean@16. Treating
them conservatively as unpaired gives a difference-SD proxy
`sqrt(.014^2+.021^2)=.02524`. A two-sided paired-t planning approximation at
alpha `.05` gives an 80%-power MDE of about `.01892` with 16 blocks (`.02243`
with 12). These are only three-seed, non-dose-matched historical proxies;
development must check whether `.030` is a credible upper bound.

## B6. Raw artifact contract

Create an atomic pair manifest plus one closed run artifact per arm.

```text
pair schema: "countdown_dose_pair_v1"
pair_state, phase, block_seed
protocol_sha256, source_lock_sha256, runtime_lock_sha256
data_manifest_sha256, sft_checkpoint_sha256, tokenizer_sha256
relabel_raw_relative_path, relabel_raw_sha256
dose_schedule[60] {completed_update, attempted_groups,
  converted_groups=d_t, relabeled_rows, schedule_record_sha256}
dose_schedule_sha256
replay_raw_relative_path, replay_raw_sha256
pair_gate_relative_path, pair_gate_sha256

run schema: "countdown_dose_run_v1"
artifact_state, provenance, config, named_rng_domains
base_batches[60]:
  completed_update, ordered task_uid[64], task_identity[64],
  generation_coordinate[64][16], original_reward_bits[64][16],
  response_text_or_external_content_address[64][16], response_sha256[64][16],
  valid_tokens[64][16], actor_before_sha256
treatment_events[60]:
  relabel: attempted_uid, admitted_uid, old/new target, rewritten prompt and
    response hashes, post-relabel rewards, verifier certificate
  replay: d_t, destination_uid, source_uid, replacement_uid,
    source/destination RNG coordinates, duplicated tensor hashes
update_events[60]:
  tensor_shape, gradient_group_count, ppo_epochs=1,
  microbatch_count, loss, grad_norm, optimizer_applied,
  actor_before_sha256, actor_after_sha256
evaluation_events[5]:
  completed_updates, tier, task_identity, task_uid,
  16 generation coordinates, 16 response texts/content addresses,
  16 binary verifier outcomes, returns/scores, entropy and token counts,
  actor_sha256
accounting:
  base_groups, train_generations, evaluation_generations,
  converted_group_exposures, tensor_rows, valid_tokens,
  optimizer_updates, wall_seconds, peak memory, checkpoint hashes
failure: null | closed failure object
```

The pair validator must prove exact equality of task order, base-generation
coordinates, batch shape, optimizer epochs, and `d_t` between arms. It must
also prove each replay destination was originally all-fail, each source was
naturally live, every replacement has a fresh 16-row UID, and no source was
selected twice within a step.

## B7. Outcome-blind gates

- exact source/runtime/data/SFT/tokenizer/verifier hashes;
- all dataset identities unique where promised and train/eval disjoint;
- recomputed tier-0/1/2 SFT overlap counts and a frozen clean tier-0 list;
- resolved config proves N=16, batch 64, LR `1e-5`, 60 updates, one epoch,
  curriculum/gate off, and identical decoding across arms;
- exact task order and coordinate-based RNG mapping across each pair;
- `d_t` equals admitted relabel groups and replay substitutions at every step;
- same 64 x 16 tensor shape, microbatch count, and optimizer opportunities;
- all relabels pass the original-task failure check, achieved-value parser,
  rewritten-task verifier, protected-answer rewrite checks, and 2-D position-ID
  reconstruction;
- replay sources are live, destinations dead, UIDs are fresh, and tensor
  copies hash exactly;
- exactly five evaluation coordinates and 128 tasks x 16 outcomes per tier;
- outcome bits recompute from retained responses and the locked verifier;
- memory <=9.0 GiB reserved in engineering, no nonfinite values, no missing
  async dump, and enough disk for 3x projected remaining writes;
- no arm aggregates, directions, CIs, or p-values in the gate.

## B8. Command status and required implementation

The historical command shape is evidenced by
`curriculum_maxrl/countdown_reviewer_arms/run_reviewer_arms.sh`:

```bash
TRAIN_SEED=1 POOL_TAG=replay_s1 ESTIMATOR=maxrl \
  CURRICULUM=false HINDSIGHT=false \
  EXTRA_ARGS="actor_rollout_ref.actor.ppo_epochs=2" \
  bash smollm/countdown_a10g.sh
```

That target script is absent, and that command is the invalid higher-dose
control. It must **not** be used for Lane B.

There is currently no honest runnable Lane-B command to print. Before launch,
the execution fork must provide and test a real entrypoint with these required
interfaces (names are TODO, not claimed existing flags):

- phase/seed/pair IDs and refuse-overwrite output root;
- mode `relabel` or `paired_live_replay`;
- required dose-schedule input hash for replay;
- exact task/SFT/source/runtime locks;
- coordinate-based task, generation, relabel, replay-slot, replay-source, and
  evaluation RNG roots;
- validation and training response dumps;
- atomic partial-state/resume semantics; and
- strict pair gate and analyzer.

The release-safe integration route is documented in
`verl_integration/README.md`. Independently review the trainer placement:
relabel after original reward computation and before old-log-prob computation.
Do not copy unlicensed upstream code into the artifact; distribute our patch
and source hashes unless the upstream license explicitly permits vendoring.

Recovery decision:

1. First try the original execution machine/backups for the exact Countdown v2
   pool, SFT checkpoint/examples, tokenizer revision, fork commits, and
   `smollm/countdown_a10g.sh`.
2. If they cannot be recovered, build and seal a **new v3 pool/SFT protocol**.
   The result can still test relabel versus replay, but it is not a direct
   confirmation of the old aggregate and must be labeled a new regime.

## B9. Expected cost and storage

The paper records roughly `6 A10G-hours` per 60-step Countdown run:

- development: 6 runs, about 36 A10G-hours;
- confirmation: 32 runs, about 192 A10G-hours;
- total after engineering: about 228 A10G-hours.

A 10-GB card with microbatching/offload may be slower; use the engineering
pair to project runtime. If a run exceeds 12 hours, the full lane exceeds 19
serial GPU-days and should move to a larger rental GPU rather than silently
reduce the protocol.

Historical planning estimated about 4.4 GB for one fp32 model+optimizer save
and about 9 GB active checkpoint storage with two retained saves. Run
sequentially, retain only the exact recovery checkpoints during execution, and
move completed checkpoints/raw responses to content-addressed external
storage. Before confirmation, budget at least 100 GB free locally and measure
compressed response-ledger size from development. Hash raw bytes before any
deterministic compression; publish both raw hash/size and compressed
hash/size.

---

# Failure recovery, integrity, anonymity, and release

## Mechanical failure recovery

- Each cell writes to a new, explicit directory and refuses overwrite.
- Write incremental events atomically. A completion receipt is created only
  after the raw artifact closes, validates, and hashes.
- A resumable checkpoint must include model, optimizer, LR scheduler,
  dataloader cursor, every RNG domain, teacher/relabel/replay state, completed
  event count, and prior-ledger hash chain. If any component is missing,
  restart the whole cell from its original seed; never splice a partial run.
- Maximum two identical retries for infrastructure failure. Preserve every
  failed attempt and incident record. A source/config change invalidates the
  lock and requires a new protocol version and fresh seed range.
- Do not replace a failed confirmation seed with a convenient new seed. Repair
  and rerun that prespecified block under identical bytes, or declare the
  confirmation incomplete.
- In Lane B, the pair is atomic. If the replay arm cannot resume exactly, rerun
  both arms and regenerate the dose schedule without reading endpoints.

## Integrity and anonymity checklist

- [ ] Immutable pre-execution protocol/commit exists and predates confirmation.
- [ ] Source lock includes transitive local Python/shell/config dependencies,
      patch files, analyzer, verifier, tests, and this final protocol.
- [ ] Runtime lock includes OS/container digest, Python, PyTorch, CUDA, driver,
      GPU model, VERL/Ray/Transformers/tokenizer versions, and deterministic
      flags. Same environment is used across arms.
- [ ] Data/SFT/model manifests contain canonical relative paths, SHA-256,
      byte sizes, row counts, and upstream revision/license.
- [ ] Duplicate-key and nonfinite JSON are rejected; schemas are closed.
- [ ] Analysis imports no runner module and recomputes every aggregate from
      raw ledgers.
- [ ] A portable verifier hashes the analyzer before import and checks the
      live runtime before reanalysis.
- [ ] Raw artifacts are append-only/content-addressed; any external payload
      has a compact receipt manifest with URI `null` if no stable anonymous
      URI exists.
- [ ] No absolute home paths, usernames, hostnames, cloud account IDs, W&B
      entities, tokens, checkpoint cache paths, Git author emails, or incident
      logs that reveal identity enter the anonymous bundle.
- [ ] Anonymous code archive contains license notices and cites upstream work
      in third person. Links do not track reviewers.
- [ ] Quick/dev/failed artifacts are clearly labeled and excluded from
      confirmatory figures and run counts.
- [ ] Paper language states any lack of a public pre-execution timestamp,
      missing external raw, pool rebuild, numerical nondeterminism, or 10-GB
      fallback honestly.

## Release boundary

Ship in the anonymous supplement:

1. protocol, source/runtime/data locks, exact commands, seed lists, closed
   schemas, tests, gates, analyzers, and analysis receipts;
2. compact per-run summaries derived from raw plus one row per scientific run
   in the generated registry;
3. raw task-level binary outcomes for every evaluation task/checkpoint;
4. full group/accounting ledgers when compact enough, otherwise a complete
   content manifest and stable anonymous external object;
5. standard pass@k recomputation and legacy-proxy warning; and
6. a one-command verifier that can verify compact artifacts without GPU and,
   when full raw is supplied, validate every response/outcome and ledger row.

Keep model checkpoints and large response text outside Git, content-addressed
and recoverable. Never publish only arm-level means.

# Handoff acceptance criteria

Lane A is ready for confirmation only when:

- the local runner passes `--help`, focused tests, strict synthetic/adversarial
  analyzer tests, and an independent pre-seal review;
- engineering completes 2/2 valid runs under the 9.0-GiB gate;
- all 18 development runs (plus any single boundary expansion) validate;
- the estimator-specific LR selection is reproduced mechanically from raw;
- the 12-block tuned/common matrix, seed order, and hashes are sealed; and
- the outcome-blind confirmation gate can be run without importing or
  emitting scientific contrasts.

Lane B is ready for confirmation only when:

- the exact v2 execution/data/SFT package is recovered, or a new v3 regime is
  explicitly frozen and renamed;
- the paired-dose controller passes synthetic adversarial tests, including
  insufficient-dead/live failures, UID collision, schedule tampering,
  wrong-reward source, tensor mismatch, and async-dump loss;
- engineering completes a valid five-step pair on the intended GPU under the
  memory gate;
- all three development pairs have exact group-dose equality and no shortage;
- the development variance/compute gate supports the frozen confirmation
  size; and
- 16 fresh block seeds, raw schema, source/runtime/data locks, and independent
  outcome-blind gate are immutable before the first confirmation rollout.

The complete research outcome is not “both hypotheses win.” It is: both
registered studies run to their frozen stopping points, all raw evidence is
retained, nulls and costs are reported, and the paper's claims follow the
registered branches without reinterpretation.
