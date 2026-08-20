# P0 — Does scoring the count law recover what the plug-in loses?

**Status:** FROZEN at 2026-08-20T20:09:15Z, before any evidence run.
Supersedes `GRANULARITY_FLIP_PREREG_v1_SUPERSEDED.md`, which never launched.
Any source change requires a new protocol/campaign and may not enter this
attempt.

**Protocol:** `group_law_flip_v1`

**Campaign:** `group-law-flip-v1-20260820-001`

**Attempt:** `attempt-001`
**Independent blocks:** 48 paired seeds, 3001–3048

## 1. Question and claim boundary

For a curriculum unit `z`, practical drop-all-fail MaxRL has count-law
coefficient activity

```text
A(z) = 2 [Pr(K>0 | z) - E(K | z)/N].
```

If a coarse unit is instead treated as one i.i.d. Bernoulli task at its mean
pass rate `p_bar`, the plug-in score is

```text
u_N(p_bar) = 2 [1 - (1-p_bar)^N - p_bar].
```

Their difference is exactly
`2[Pr(K=0|z) - (1-p_bar)^N]`. Its nonnegative sign is guaranteed for the
paper's mixture-of-conditionally-i.i.d.-tasks aggregation regime, not for an
arbitrary count law. This Tier-1 identity is already proved and checked. The
open question is causal and substrate-specific: with everything else fixed,
does using `A(z)` rather than `u_N(p_bar)` improve learning in MAZE-SCORE?

This study can support causal relevance of the correction on this substrate.
It cannot show that the identity predicted a downstream sign, that the
correction solely mediated the historical MAZE-SCORE result, that either arm
beats `p(1-p)`, or that the result transfers to another partition/model/budget.

## 2. Intervention: one functional changes

Both arms use the same four-moment count-law posterior, deterministic posterior
mean, prior, decay, uniform floor, categorical sampler, estimator, and group
observations available within that arm. Only the score functional changes.

| field | `plugin` | `grouplaw` |
|---|---|---|
| teacher | `group_law_plugin` | `group_law_activity` |
| posterior state | `(W,Z,S,S2)` per level | same |
| posterior statistic | mean count law | same |
| raw score | `u_32(E[K]/32)` | `2(1-Z/W-S/(32W))` |
| score metadata | `iid_plugin_from_count_law_mean`, exponent 32 | `group_law_activity`, exponent null |

Frozen shared posterior/teacher settings:

- `N=32`, `p0=.5`, prior mass `2/N=.0625` pseudo-groups (exactly two
  pseudo-rollouts, matching the strength of Beta(1,1));
- excess-evidence decay `.7`, applied after every closed group;
- deterministic posterior mean, with no Thompson draw;
- nonnegative score normalization followed by a `.15` uniform floor;
- eight sampled levels/groups per update.

The arms agree exactly before data. They separate only when observed success
counts make the level count law non-binomial at its mean. Live visits then
diverge by design; “same posterior” means the state definition and update rule
are identical, not that the two trained policies later see identical groups.

## 3. Frozen substrate and pairing

This inherits MAZE-SCORE v2 unchanged:

- one 1.26M-parameter maze Transformer (`d_model=128`, 6 layers), MaxRL,
  AdamW learning rate `1e-4`;
- one seed-specific 600-step SFT warmstart shared byte-for-byte by the two arms;
- 250 RL updates, 8 tasks/update, 32 rollouts/task;
- held-out evaluation at post-SFT update 0 and completed updates
  25, 50, ..., 250;
- 32 held-out tasks/level and 8 samples/task over the same 13 levels;
- no hindsight, dense hindsight, teacher relabeling, or wall-clock stopping;
- seed block fixes model/SFT/RL streams; evaluation-task seed is
  `202608130+seed`, evaluation-sample seed is `302608130+seed`, and teacher
  seed is `seed+77`;
- per-level task RNGs and per-update rollout resets preserve the existing
  paired-arm randomization contract.

Process order is counterbalanced: even seeds run `plugin` then `grouplaw`; odd
seeds run the reverse (24 blocks in each order). The seed block, not an update,
level, group, or rollout, is the independent unit.

## 4. Outcome-blind design calibration

No P0 result existed when the following decisions were made.

### Power and block count

`GROUP_LAW_FLIP_POWER_MEMO_2026-08-20.md` simulates the full support
conjunction using the historical paired-SD range `.0077–.0135`, with a paired
t test/interval proxy for the frozen exact/bootstrapped tests. At pessimistic
SD `.0135`, 48 blocks give estimated power `.901` for the declared
powered-for effect `+.0075`, compared with `.653` for the superseded 20-block
draft. A true effect exactly at the `+.005` SESOI has only about `.503` support
probability because the observed point estimate must itself reach the SESOI.

Seeds are therefore 3001–3048. No seed may be added after any P0 run begins.
The two 24-seed Slurm submissions are scheduling chunks, not interim looks.

### Treatment-delivery calibration

`GROUP_LAW_FLIP_DELIVERY_REPLAY_2026-08-20.json` replays only historical
MAZE-SCORE `un`-arm selected levels and group success counts through both
shared-posterior functionals. It never opens an evaluation record. Across 48
blocks, the mean TV between the two expected full-run visit distributions is
`.29959` (block range `.24960–.34901`); mean update-level TV is `.33490`.
This is threshold calibration, not a counterfactual endpoint prediction,
because changed visits would change both later posterior states and policies.

## 5. Endpoints and frozen decision rule

### Treatment-delivery gate

For each completed block and arm, form the empirical distribution of all 2,000
selected levels (250 updates × 8 groups). Compute total variation between the
two arm distributions, then average the 48 block TVs. The gate passes iff the
mean is **at least `.05`**.

If it fails, the primary is still calculated and archived but its registered
decision is `treatment_not_delivered`, never evidence against the coefficient
identity or against a sufficiently separated intervention.

### Primary endpoint

For each arm and block:

```text
cov_auc_delta = mean(mean-level pass@8 coverage at updates 25..250)
                - mean-level post-SFT pass@8 coverage.
```

The sole primary contrast is paired `grouplaw - plugin`. Exactly ten RL
timepoints enter. Missing, duplicate, extra, or nonterminal timepoints refuse
the whole campaign.

Support requires all four conditions:

1. treatment-delivery gate passes;
2. mean paired difference is at least `+.005` cov-AUC (SESOI);
3. paired percentile-bootstrap 95% CI lower bound is greater than zero
   (20,000 resamples; seed 20260820; NumPy linear percentile);
4. exact two-sided paired sign-flip `p <= .05`.

`practically_ruled_out` requires delivery to pass, support to fail, and the CI
upper bound to be below `+.005`. All other delivered results are
`inconclusive`. There is one confirmatory contrast, hence no multiplicity
adjustment. Positive/negative/zero pair counts are descriptive.

### Frozen descriptive secondary

Pool observed P0 group counts across arms and blocks within each level and
report
`2[P_hat(K=0|z) - (1-p_bar_z)^32]`. Also report its Spearman rank correlation
with the per-level paired `grouplaw-plugin` cov-AUC difference. This secondary
has no decision threshold and cannot promote the primary verdict.

## 6. Missingness, retries, and outcome firewall

The analyzer requires a complete 2×48 matrix. It never performs available-case
analysis or seed substitution.

Each Slurm job writes to a job-specific quarantine directory outside the
attempt root. Only after both arms reach 250 updates, their result/telemetry
loaders pass, both terminal checkpoints exist, receipts are written, and the
block SHA256 manifest verifies is the whole directory atomically moved to
`attempts/attempt-001/seed-{seed}`. A nonzero/infrastructure failure may be
retried only while no final block exists; every quarantined attempt is retained
and excluded. The retry reruns the full paired block. Completed blocks are
never overwritten. Scientific-result-driven retries, arm-only retries, seed
replacement, and extra seeds are forbidden.

Humans and progress monitors may inspect scheduler state, process exit, file
counts, receipt presence, hashes, and telemetry integrity, but not result JSONL
contents or endpoint summaries. The analysis loader validates all block
manifests, source identity, receipts, configs, shared warmstarts, terminal
checkpoints, 96 run files' evaluation schedules, and 24,000 telemetry rows before
calculating a campaign endpoint. Progress output contains no endpoint value.

The command-line analyzer creates an exclusive claim inside the immutable
local campaign and refuses a second invocation or an existing output. Its
output must live outside the campaign directory.

## 7. Prospective verdict language

**Supported:** “In the preregistered MAZE-SCORE intervention, replacing the
i.i.d.-at-the-mean plug-in with the shared-posterior count-law functional
improved cov-AUC by Δ (95% CI [...], exact paired p=..., k/48 positive blocks),
with delivered visit TV .... This establishes causal relevance of the
count-law correction on this substrate. It does not show that the algebra
predicted the learning sign or that this mechanism alone explains the earlier
MAZE-SCORE contrast.” Tier 2.

**Practically ruled out:** “The delivered count-law intervention did not reach
the preregistered `+.005` learning SESOI (Δ ..., 95% CI [...], visit TV ...).
The exact coefficient-calibration identity therefore remains Tier 1, while a
material learning consequence on this substrate and budget is bounded.” Tier
3.

**Inconclusive:** Report Δ, CI, exact p, sign count, and visit TV; retain the
learning claim as open. Do not add seeds or promote the descriptive secondary.

**Treatment not delivered:** Report the failed TV gate and label the endpoint
non-diagnostic. Do not interpret its sign.

No branch permits claiming superiority to `p(1-p)` because that arm is absent.

## 8. Immutable artifacts and paths

Evidence source files:

- trainer: `curriculum_maxrl/maze_gpu/train.py`
- count-law sufficient statistics: `curriculum_maxrl/count_law_stats.py`
- analyzer: `curriculum_maxrl/group_law_flip/analyze_group_law_flip.py`
- launcher: `hopper/sbatch/group_law_flip_array.sbatch`
- environment lock: `hopper/requirements-maze-hopper.lock`

Remote final blocks:
`/scratch/lwang44/maxrl/group_law_flip/campaigns/group-law-flip-v1-20260820-001/attempts/attempt-001/seed-{3001..3048}`.

Canonical local immutable campaign after retrieval:
`/data/robotixx/group_law_flip/group-law-flip-v1-20260820-001/attempt-001`.

Single-use analysis output:
`curriculum_maxrl/group_law_flip/GROUP_LAW_FLIP_ANALYSIS.json`.

Frozen file/environment identities:

| item | SHA256 / identity |
|---|---|
| trainer | `bab28a5c43eb7e886feb0bf1c22983063fca4f199ad0b47c664d83679a5d158d` |
| count-law statistics | `29a9df9bce569603162598c0e78a64d89615699fe4a4d51de55495205cad3515` |
| analyzer | `9d88d6d5e63110eb1f1fa292f88a4d6c8b33d4878ef6fbf803c3bf5272186603` |
| evidence launcher | `02176cf66d49791d4fdefafcd5f379d844f0862d6a91ffd12f25eb45b598fc3a` |
| power memo | `01e71c8349f957d76cce0d3c4143fcaa45aa4b96f15737ddc307fa742af4d656` |
| outcome-blind delivery replay | `574937231a75c7a0dda3fed60c2d0c86dd0e5682b12d3a46b08b370456af366a` |
| environment lock | `ad774d459fa77bb68c01c4a225db1e7faa3213216422eb5eabdf5b3c0e3d6224` |
| environment freeze | `70d7f2c337b75de70adf941dacefdb7d3f7ba1772ac7f32821c896a61e77f36a` |
| environment JSON | `42efa0bf38cc6d4aca56eac21559dfc989c92abda49e9eaf5df4fbcf019bf393` |
| successful engineering smoke | Hopper Slurm job `9419940`, exit 0, 29 seconds |

The earlier engineering-only smoke job `9419891` failed before RL because its
legacy-mode harness passed an unsupported absolute warmstart path. The harness
was corrected to run from a disposable bundle copy; no evidence data existed,
and the successful job above exercised both teachers, telemetry, and terminal
checkpoint writes.

The freeze git commit and evidence bundle manifest are necessarily external
bindings (neither a commit nor a manifest can contain its own final hash).
`SOURCE_STATE.json` in the content-addressed evidence bundle binds the commit;
the post-staging launch receipt records both identities before submission.
