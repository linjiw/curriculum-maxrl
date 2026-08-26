# E4: frozen-checkpoint LLM count-law calibration

**Status:** DRAFT v1 — NOT FROZEN; NOT AUTHORIZED TO SAMPLE. The PI must
approve this record and freeze it with the generation/analyzer hashes in a
clean commit no later than 2026-09-03. If that does not happen, E4 is dropped.
All E4 work stops on 2026-09-08; there is no extension or replacement
experiment.

## 1. Scope and evidence tier

E4 is one training-free measurement on one frozen small language model. It
asks whether the same-mean/coarse-pooling calibration gap measured elsewhere
is visible when a practical LLM task pool is aggregated into operand-count
buckets. It does not train a policy, compare curricula, test learning utility,
or reopen the de-scoped LLM RL experiment.

Any complete result is **Tier 2′ controlled but descriptive**. It will be
reported as a measurement, never as "confirmed." There is no hypothesis test,
p-value, efficacy verdict, sign gate, SESOI, or decision rule. Consequently
there is no outcome branch to write. The result is the preregistered vector of
bucket measurements, including negative or near-zero values if observed.

## 2. Frozen substrate

### Model

- Architecture/source: SmolLM2-360M-Instruct, base revision
  `a10cc1512eabd3dde888204e902eca88bddb4951`, followed by the already completed
  one-epoch clean Countdown SFT recorded in `training_metrics.json`.
- Evaluation checkpoint: the existing immutable directory conventionally
  named `countdown_sft_clean_v1`; no weights, adapters, prompts, or tokenizer
  files may change for E4.
- Checkpoint identity is the `model.safetensors` SHA-256
  `3198bb0f0c8598ec9aa713540e19472ebbe8363702db0d555fb060c679128ff8`.
- Supporting locks: `config.json`
  `283834b57c6e55af57e59b007df3bfcaf2f898dbb22fb535a46d224b73acb0cd`;
  `tokenizer.json`
  `7d27c493c729a66ecefc837280b05d948b1ed50d130eebdbf911b1b36cf38ed7`;
  `training_metrics.json`
  `36085b432f5d3bed12e192648e996de6a10c41a60f9955582cce56b5bd8589f4`;
  `generation_config.json`
  `a490a332c7ef056bdc19531cf62d517cf9d24a589a1a989c6259e8dac1002ace`;
  and `chat_template.jinja`
  `872be49dbb638044ad01b60388f48d469ff2980e5f0dccdc22ec907db54d0788`.

This is a frozen checkpoint, not a frozen-checkpoint *smoke* for an RL run.
No optimizer is instantiated and no gradient or parameter update is allowed.

### Tasks and coarse units

- Dataset: all 384 rows of the existing held-out Countdown v2
  `test.parquet`, SHA-256
  `95b1456fc3f49bc6f463614fef92900d748a07e4b429fd9383bbcf5edcb4e489`.
- Dataset manifest SHA-256:
  `3853be854ee3dcf8ba713ac1a9550da9131d7bd954b93e70afad7d5c578134aa`.
  It records zero train/test and zero SFT/test task overlap.
- Atomic task identity: `(integer target, sorted integer operand multiset)`.
- Coarse unit `z`: operand count, fixed prospectively as two, three, or four
  operands (`countdown_tier0`, `countdown_tier1`, `countdown_tier2`).
- Sample size: every one of the 128 held-out atomic tasks in each bucket;
  384 tasks total. There is no task subsampling or replacement.
- One atomic task is shared within each rollout group. A group contains
  `N=16` stochastic completions of that same prompt. Thus the run produces an
  exact `3 x 128 x 16 = 6,144` binary-outcome matrix.

The estimator defines the coefficient map; the curriculum defines the unit
over which that map is averaged. These operations do not commute.

## 3. Generation protocol

The frozen evaluator is `curriculum_maxrl/countdown/eval_countdown.py`, current
SHA-256
`0f642db64cabff66631b7e9ac88f1f3519651b21bee351051a1190f1a5bf653d`.
The strict binary verifier is
`curriculum_maxrl/countdown/countdown_reward.py`, current SHA-256
`99c04d4a4914170a528c67337aec364e7410074c552d9848c714f78c0f9e2312`.
Both hashes must be rebound to the freeze commit and checked immediately
before generation.

Frozen generation settings:

- dataset order: the committed/parquet row order, all rows;
- prompt batch size: 8;
- completions per task: 16;
- maximum new tokens: 128;
- stochastic decoding: enabled;
- temperature: 1.0;
- top-p: 1.0;
- model dtype: bfloat16;
- generation seed: 20260903, reset after model loading as the evaluator does;
- tokenizer chat template: the locked checkpoint template;
- raw retention: completion text, achieved value, binary reward, and new-token
  count for every rollout.

The intended command is:

```bash
python -m curriculum_maxrl.countdown.eval_countdown \
  --model "$FROZEN_E4_MODEL" \
  --data "$FROZEN_E4_TEST_PARQUET" \
  --output "$NEW_E4_OUTPUT_SUMMARY" \
  --k 16 --batch-size 8 --max-new-tokens 128 \
  --temperature 1.0 --top-p 1.0 --seed 20260903
```

The freeze commit must add a fail-closed preflight wrapper that substitutes
the exact operational paths, rejects any hash or setting mismatch, requires a
new empty output directory, and records the environment. Direct generation is
not authorized from this draft.

## 4. Preregistered statistic

For task `i` in bucket `z`, let `K_iz` be its count of successes among 16
completions. Define

```text
p_bar_z = (1 / (128 * 16)) sum_i K_iz
q0_z    = (1 / 128) sum_i 1{K_iz = 0}
G_z     = q0_z - (1 - p_bar_z)^16
```

The sole preregistered report is the ordered three-vector
`(G_2-operands, G_3-operands, G_4-operands)`. For interpretation, the same
table also reports:

```text
A_count,z  = 2 (1 - q0_z - p_bar_z)
A_plugin,z = 2 (1 - p_bar_z - (1 - p_bar_z)^16)
A_plugin,z - A_count,z = 2 G_z.
```

These are empirical count-law and plug-in coefficient activities for the
fixed held-out cohort. They are not gradient norms, learning progress, policy
improvement, or evidence of endpoint mediation. The population nonnegative
aggregation-gap corollary requires a task-shared, conditionally-i.i.d. mixture;
finite-cohort estimates and non-i.i.d. model sampling need not be nonnegative.

Per bucket the appendix table will additionally show `p_bar_z`, `q0_z`,
`(1-p_bar_z)^16`, observed-set pass@16 `1-q0_z`, and the full histogram of
`K=0,...,16`. Standard pass@16 is valid here because every raw binary outcome
is retained.

### Descriptive uncertainty

For each bucket separately, resample its 128 atomic-task rows with replacement
10,000 times, recompute `G_z`, and report the 2.5th and 97.5th type-7
percentiles as a **task-bootstrap 95% interval**. The fixed bootstrap seed is
20260903 plus the zero-based lexicographic bucket offset. The interval is
descriptive; it is not a test and does not determine whether E4 "passes."

The draft analyzer is
`llm_calibration/analyze_count_law_calibration.py`. It validates the complete
matrix, refuses to overwrite an existing analysis, emits no p-value or verdict,
and must be hash-bound in the freeze commit. Its current unit tests are
`llm_calibration/test_analyze_count_law_calibration.py`.

## 5. Completeness and failure handling

Analysis is allowed only if all of the following are true:

1. all locked checkpoint, tokenizer, dataset, evaluator, verifier, analyzer,
   and environment fingerprints match the freeze record;
2. exactly 128 unique atomic tasks exist in each of the three fixed buckets;
3. every task has exactly 16 retained completion/reward records;
4. all rewards are binary and operand count agrees with the bucket;
5. the output paths were absent before generation and the analysis path is
   absent before the single analysis; and
6. no model parameter was updated.

An interruption may resume only missing task groups under a written,
outcome-blind engineering receipt that preserves completed raw groups and the
frozen settings. No completed task is resampled, no task is substituted, and
no endpoint is inspected to decide whether to resume. If the complete matrix
is not available by 2026-09-08, E4 ends without a result.

## 6. Reporting perimeter

If complete by the hard stop, E4 may add exactly one descriptive appendix
figure/table and one sentence in Section 3.4. Permitted wording is:

> On one frozen 360M-parameter Countdown checkpoint, the preregistered
> descriptive measurement found bucketwise all-fail plug-in gaps of [report
> all three values with task-bootstrap 95% intervals]. This describes
> coarse-unit calibration on that fixed pool; it is not evidence of mediation
> or learning utility.

The bracket is filled mechanically with no sign-contingent branch. No abstract,
contribution, conclusion, README headline, or website headline changes. A
complete result is a calibration measurement at LLM scale, not an LLM
training result. An incomplete run creates no evidence and is recorded only in
an operational closure note.

## 7. Freeze checklist

Before the PI changes the status to FROZEN and before any model sampling:

- [ ] approve the fixed checkpoint, three operand-count buckets, all 384
      tasks, N=16, settings, statistic, and reporting perimeter;
- [ ] add and test the fail-closed generation preflight/launcher;
- [ ] record the exact runtime environment and CUDA/GPU type;
- [ ] run the analyzer tests and generation dry tests without model sampling;
- [ ] bind the evaluator, verifier, analyzer, launcher, data, and checkpoint
      hashes in this file;
- [ ] commit the complete record from a clean tree by 2026-09-03; and
- [ ] record the freeze commit before generation begins.

Until every box is satisfied, this remains an internal draft and sampling is
forbidden.
