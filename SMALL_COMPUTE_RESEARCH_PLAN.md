# Small-compute research plan for the ICLR paper

Updated 2026-08-09. All Mac experiments required for the current submission
are complete. This plan now separates submission-critical evidence recovery
from post-submission Mac and <=10-GB GPU studies. Nothing below is a new
confirmatory claim unless a protocol is frozen before the corresponding runs.

## What the closest literature changes

The generic claim "sample intermediate-difficulty prompts" is crowded. The
paper should claim the exact practical-MaxRL coefficient-activity convention
and report joint estimator/sampler tests with their mixed outcomes; the fresh
Digits factorial does not establish a general estimator-by-sampler
interaction.

- [ProCuRL](https://arxiv.org/abs/2304.12877) gives the target-success-one
  score `p(1-p)`; our expected half-mass equals this only at `N=2`.
- [GRESO](https://arxiv.org/abs/2506.02177) and
  [DPS](https://arxiv.org/abs/2603.10887) select prompts before rollout from
  historical or predicted reward dynamics.
- [VIP](https://arxiv.org/abs/2602.01601),
  [HORA](https://arxiv.org/abs/2605.07114), and
  [VIGOR](https://arxiv.org/abs/2607.22002) vary the number of rollouts across
  prompts using predicted gradient variance, posterior hit utility, or
  realized variance.
- [BBG](https://arxiv.org/abs/2606.15455) explicitly optimizes a posterior
  marginal pass@k contribution, while
  [Curriculum RL](https://arxiv.org/abs/2606.22317) adds teacher guidance near
  or beyond an estimated reasoning boundary.
- [RL2ML](https://arxiv.org/abs/2605.30154) makes the estimator convention
  crucial: its `T=N` alignment retains the score-baseline control variate on
  all-fail groups, whereas our deployed hybrid centers only when `K>0` and
  zeros the entire `K=0` group, giving `T=N-1`.
- SPEED-RL ([arXiv:2506.09016](https://arxiv.org/abs/2506.09016)) is withdrawn
  because its authors report unresolved experimental bugs. It is excluded as
  empirical support and as a required comparator.

## Work already completed on the Mac

### Fixed-completion N-sweep

Across `N={2,4,8,16,32}`, eight paired seeds, and exactly 51,200 completions
per cell, `u_N` beats the score-only `p(1-p)` comparator in 8/8 seeds for
every `N>2`. It does not universally beat uniform. This is a useful
post-guidance mechanism check, not an algorithm benchmark or scaling law.

### Exact-probability estimator × sampler factorial — complete, negative

A source-locked CPU contextual-bandit study crossed practical MaxRL and RLOO
with uniform, `p(1-p)`, and `u_8` sampling on `sklearn` Digits. Exact
correct-class probabilities remove posterior and probe error. Five-rate
development selected learning rate `.1` for both estimators and the common
schedule. All 24 fresh paired confirmation blocks completed.

The registered interaction was `+0.01589` with bootstrap 95% interval
`[-0.01686,+0.04712]` and exact sign-flip `p=.350`, so it was not supported.
MaxRL favored `u_8` over `p(1-p)` by `+.20842`, but RLOO also favored `u_8`,
reversing its registered prediction. The estimator-matched sampler was below
uniform for both MaxRL (`-.11279`) and RLOO (`-.37581`). On the retained local
full payload, the tuned and common ledgers/checkpoints are byte-identical, and
their scientific summary content
matches after removing phase/authorization labels, because every selected
rate is `.1`; they are an identity check, not independent replications. This
closes
the proposed exact-`p` interaction test as a negative result and strengthens
the paper's caution that coefficient mass measures activity rather than
universal curriculum value. Full gates, budgets, effects, hashes, and claim
boundaries are in the linked result. The compact repository contains the
per-block contrasts and a 2,904-file content manifest; the 5.08 GB historical
replay payload is not yet downloadable from a clean clone.
[`curriculum_maxrl/digits_factorial/RESULTS.md`](curriculum_maxrl/digits_factorial/RESULTS.md).

### HORA allocation factorial

A 2-sampler by 3-allocation CPU factorial compares fixed `N=16`,
published-HORA-style posterior hit allocation, and a mass-aware marginal that
counts four already-spent probes. It uses 16 paired seeds and 51,200
completions per cell.

- Mass-aware minus HORA-style pass@8 AUC: `+0.00676` (11/16 positive).
- Mass-aware minus fixed pass@8 AUC: `+0.02517` (14/16 positive).
- Mass-aware minus HORA-style realized coefficient mass per completion:
  `-0.000306` (0/16 positive).

The performance direction is interesting, but the frozen coefficient-mass
mechanism is falsified. The allocator also creates extreme groups (average
cell maximum about 76). This belongs in the appendix and motivates caps and
posterior calibration; it is not a fourth contribution.

### Correlated-rollout mass stress test

The distribution-free identity
`E[sum_i |w_i|] = 2(P(K>=1)-E[K]/N)` survives without conditional
independence; with a common marginal it reduces to `2(P(K>=1)-p)`.
An exact beta-binomial sweep over `N={2,4,8,16,32}` and positive within-group
correlations verifies the identity to `2.58e-13`, but shows that the i.i.d.
substitution can be optimistic. At `rho=.10`, the activity-maximizing `p`
moves from `.169` to `.234` for `N=16`, and from `.106` to `.186` for `N=32`.
This strengthens the theorem's scope statement without claiming a learning
benefit or that beta-binomial dependence describes every rollout process.

### Frozen AUC robustness analysis

The wave-2 multiverse is implemented for simple versus trapezoidal AUC,
warm-start inclusion, early/mid/full horizons, and leave-one-checkpoint-out
variants. It correctly exits as blocked because all 24 required checkpoint
trajectories remain outside this checkout. Scalar AUC summaries cannot
identify those alternatives.

### Paid-probe ProCuRL selection attachment — complete, primary unsupported

The separately frozen four-arm attachment completed all 12 development runs
and all 320/320 confirmation runs. The registered `u_16-ProCuRL` fixed-paid-AUC
contrast was `+.004894`, with `t(79)=1.9773` and `p=.05149`; it fell below the
`.02` SESOI and is unsupported. Each probed arm spent about 93.2% of its paid
transitions on probes. Ordinary-uniform fixed-paid AUC was `.65149`, versus
`.33771` for ProCuRL-env, `.33942` for probe-sham uniform, and `.34261` for
`u_16`.

The safe conclusion is narrow: probe cost dominated this actor-only,
fixed-pool attachment at the frozen refresh cadence. The result does not show
that full PPO ProCuRL is inferior, nor does it predict performance under a
cheaper cadence. The compact release includes receipts and a content manifest;
the 1,374,886,097-byte raw ledger remains external with SHA-256
`b1f8756c249effab8c77101c8bca73ddf708a5e143c18fe8742fd5712fdd7c12`.
The execution is internally bound by source/runtime/gate/artifact hashes, but
there is no immutable public pre-execution commit. The registry generator emits
and checks exactly 562 records, including 441 Acrobot records, and owns the exact
totals. See
[`ACROBOT_PROCURL_SELECTION_RESULTS.md`](frontier_rl/examples/ACROBOT_PROCURL_SELECTION_RESULTS.md).

## Priority 0: recover evidence, no GPU

This has greater reviewer value than another experiment.

An exhaustive 2026-08-08 local audit is complete and found no surviving copy
in any Git/ref/reflog/unreachable object, LFS object, worktree, stash, local
backup, sibling checkout, Ray/cache path, or home-directory signature match.
The items below now require the original EC2/EBS/S3/W&B side; otherwise they
must be rerun. See `curriculum_maxrl/analysis/ARTIFACT_RECOVERY_AUDIT.md`.

1. Import the 24 wave-2 checkpoint JSONLs and warm starts from execution fork
   `9f7dd2e`, preserving hashes and source paths.
2. Recover complete Countdown B1/B2 records, especially seed 3, plus the 16
   binary verifier outcomes for every task, arm, and seed.
3. Recover the SFT and evaluation task manifests so the 27/128 tier-0 overlap
   count can be recomputed rather than trusted from a stored summary.
4. Vendor the external registration-lock objects or a signed/timestamped
   bundle if they still exist.
5. Run the frozen maze AUC multiverse and recompute standard unbiased pass@k
   from Countdown task outcomes.

The last item is essential because the logged Countdown field currently
called `pass@16` is VERL's 1,000-resample, with-replacement `best@16` proxy.
It is not the standard unbiased pass@16 estimator.

## Post-submission Mac follow-ups (deferred)

These are research opportunities, not missing experiments for the current
submission. PLR, PAIRED, ACCEL, and full SFL comparisons are likewise deferred
to separately frozen post-submission studies.

### 1. Capped and calibrated HORA factorial — complete

The frozen 50-cell by 16-seed matrix completed with all 800 runs independently
validated.  The prospective engineering filter selected cap 32: in the
fresh-group-proxy/same-step cells averaged over samplers, it reduced mean
per-run maximum group size by 58.07% while changing pass@8 AUC by -0.00271
relative to uncapped.  Several cap-32 adaptive cells improve descriptive AUC
over the fixed anchor, but allocator and information-source contrasts are
heterogeneous, and positive learning contrasts still coexist with lower
realized coefficient mass.  This does not rescue the mediation hypothesis or
create a new confirmed contribution.  Exact validation, hashes, all registered
contrast-family counts, and the claim boundary are in
[`curriculum_maxrl/CAPPED_HORA_ROBUSTNESS_RESULTS.md`](curriculum_maxrl/CAPPED_HORA_ROBUSTNESS_RESULTS.md).

### 2. Offline next-window predictive audit — blocked/deferred

From recovered trajectories, predict the next evaluation-window improvement
using `u_N`, `p(1-p)`, realized variance, learning progress, loss, and a BBG-like
boundary utility. Split by seed block or generated topology, never by
checkpoint, to avoid trajectory leakage. Report out-of-block rank correlation
and calibration rather than selecting the best predictor on the same data.

### 3. Correlated-rollout stress test — complete

The exact sweep, count-distribution verification, fixed-seed Monte Carlo
audit, structured artifact, and reproduction tests are retained in
`curriculum_maxrl/CORRELATED_ROLLOUT_STRESS.md`. A future empirical extension
would estimate within-prompt outcome correlation from recovered neural logs;
that extension remains blocked by the missing trajectories.

### 4. Cheap baseline tournament — deferred

On several generated skill-chain topologies, compare fixed-N cross-prompt
selection by uniform, `u_N`, `p(1-p)`, GRESO-like historical dynamics, and
DPS-like predicted dynamics under an identical completion budget. Keep
topology as the independent unit. Separately compare variable-N allocation by
fixed, HORA, VIP-like variance prediction, VIGOR-like realized variance, and
the capped mass-aware rule. Do not mix cross-prompt selection and variable-N
allocation into one undifferentiated leaderboard.

## Post-submission experiments for a GPU with at most 10 GB

### A. Estimator-specific maze learning-rate calibration

This addresses the largest threat to the current maze result: MaxRL and GRPO
share one deployed learning rate rather than receiving estimator-specific
tuning. Use the existing 1.26M-parameter maze model.

1. On development seeds only, run a small log-scale learning-rate grid for
   each estimator under both samplers.
2. Select one rate per estimator using a frozen development criterion.
3. Confirm the selected pair on fresh shared seed/warm-start blocks, keeping
   `N=32`, group count, training steps, and evaluation schedule unchanged.
4. Report both the common-rate and tuned-rate factorials; do not replace the
   original result.

This model is small enough that the limiting resource should be runtime, not
10 GB memory, and it directly tests whether the coverage ordering is an
optimizer-scale artifact.

### B. Truly dose-matched Countdown recycling control

Use a smaller model such as SmolLM2-135M with LoRA or another memory-bounded
adapter setup. Run a non-evidence memory/treatment pilot first, then freeze:

- no auxiliary update;
- exact-verifier relabel update;
- live-group replay update matched to relabeling in auxiliary group count,
  optimizer steps, and trained tokens.

Use three seeds, disjoint task pools, the same step budget, and a fixed tier-1
evaluation set. Store all 16 binary outcomes per task and compute both mean@16
and standard unbiased pass@k. The relabel-versus-live contrast estimates the
direction effect at matched dose; each auxiliary arm versus baseline estimates
the package effect. Monitor peak allocated memory and stop the pilot if it
cannot stay below the machine's safe limit.

## Decision rule for submission

- **Stop new Mac execution.** The submission experiment set is complete.
- Recover central raw trajectories/outcomes if external storage becomes
  available; this is artifact recovery, not a reason to launch substitute
  local experiments.
- Preserve the completed compact paper/artifact rebuild and generated registry,
  whose exact totals are 562 records and 441 Acrobot records.
- Treat estimator-specific maze calibration, dose-matched Countdown, PLR,
  PAIRED, ACCEL, and full SFL as post-submission work, not conditions for
  finishing the present draft.
