# GROUP-LAW-FLIP result — count-law correction is causally relevant on MAZE-SCORE

**Campaign:** `group-law-flip-v1-20260820-001`, `attempt-001`, 48 paired
blocks (seeds 3001–3048), protocol `group_law_flip_v1`, source manifest
`b0cf3d2deb388a8f3eac7c05d15c14dbe8e3f6cda58b8a1903e27f31f3a2c95a`.
**Preregistration:** `granularity_flip/GROUP_LAW_FLIP_PREREG.md`, frozen in
commit `f27ba8a` before evidence launch and unmodified thereafter.
**Analyzer:** `curriculum_maxrl/group_law_flip/analyze_group_law_flip.py`,
SHA-256 `9d88d6d5e63110eb1f1fa292f88a4d6c8b33d4878ef6fbf803c3bf5272186603`,
matching the frozen hash. It was run **once** after pre-unblinding commit
`8349888`.
**Analysis artifact:**
`curriculum_maxrl/group_law_flip/GROUP_LAW_FLIP_ANALYSIS.json`, SHA-256
`c1e6dc3ead1ef11db90fa2380999c3f3c45d5bc5c8c8fb234420632bc0d952e9`.

All 48 evidence allocations were terminal `COMPLETED 0:0`; the dependency
helper was `COMPLETED 0:0`. Before retrieval, the closure verified 48 final
blocks, 48 `COMPLETE` markers, 48 block manifests, 96 arm receipts, zero
invalid blocks, and zero incomplete quarantines. Every remote block manifest
passed before the complete campaign was fetched to
`/data/robotixx/group_law_flip/group-law-flip-v1-20260820-001/attempt-001`.
The frozen loader then validated the complete 2×48 matrix, source and
warmstart pairing, terminal checkpoints, exactly ten RL evaluation points,
and 24,000 telemetry rows before calculating an endpoint. No endpoint was
opened before this closure.

## Primary: supported

Endpoint `cov_auc_delta` is mean level-coverage pass@8 over the ten completed
RL evaluations at updates 25–250 minus post-SFT coverage, paired within seed
block. The sole primary contrast is `grouplaw - plugin`.

| contrast | mean | 95% paired-bootstrap CI | exact sign-flip p | pairs | SESOI | decision |
|---|---:|---:|---:|---:|---:|---|
| **`grouplaw - plugin`** | **+.00666** | **[+.00441, +.00887]** | **9.56e-7** | **40+/8−/0=** | **+.005** | **supported** |

All four frozen support conditions passed: the treatment was delivered, the
observed mean exceeded the `+.005` SESOI, the interval lower bound was above
zero, and the exact two-sided paired sign-flip p-value was below `.05`. No
seeds were added, no endpoint was substituted, and neither arm was rerun.

## Treatment delivery: passed

The preregistered delivery metric is the mean, across blocks, of total
variation between the two arms' empirical full-run level-visit distributions.
It was **.33597**, above the frozen `.05` threshold. Thus the result is not a
delivery-gated null or a comparison between statistically indistinguishable
samplers.

## Registered interpretation and boundary

In the preregistered MAZE-SCORE intervention, replacing the
i.i.d.-at-the-mean plug-in with the shared-posterior count-law functional
improved cov-AUC by `+.00666` (95% CI `[+.00441,+.00887]`, exact paired
`p=9.56e-7`, 40/48 positive blocks), with delivered visit TV `.33597`. This
establishes causal relevance of the count-law correction on this substrate.
It does **not** show that Corollary 2 predicted the learning sign, that the
correction alone explains the earlier MAZE-SCORE contrast, that either arm
beats `p(1-p)`, or that the result transfers to another partition, model, or
budget.

The exact count-law and granularity identities remain Tier 1. This completed
intervention is Tier 2: preregistered and confirmed on the MAZE-SCORE
substrate.

## Descriptive secondary

The frozen descriptive secondary pools observed group counts across both arms
and blocks. Its per-level coefficient-activity correction has Spearman
correlation `.157` with the per-level coverage contrast. This quantity had no
decision threshold and is descriptive only; it is not promoted as mediation
evidence.

## Consequence for the paper

The earlier MAZE-SCORE result remains a Tier-3 negative for sampling by
`u_32` rather than `p(1-p)`. P0 isolates a different question and answers it:
when the curriculum unit is a pooled level, using its observed count law rather
than pretending the level is one Bernoulli task yields a material learning
gain under the frozen budget. The paper can therefore recommend scoring the
count law on this substrate while retaining its central boundary: even exact
coefficient activity is not a universal measure of learning utility.
