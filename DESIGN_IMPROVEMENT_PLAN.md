# Design improvement plan

**Review date:** 2026-08-14 (America/New_York)

**Primary target:** ICLR 2027 (`paper/body_iclr.tex` via `paper/main_iclr2027.tex`)

**Secondary track:** ICRA 2027 navigation, kept scientifically separate

**Audit basis:** current manuscript and source, checked-in analyses and verdicts,
live E2c readiness receipt, Hopper job registry, a fresh compile, the repository's
reproduction command, and the closest current RLVR curriculum literature.

## Executive verdict

There is a defensible paper here, but the current version is still a borderline
main-conference submission and is `HOLD` as a competitive method paper. The
exact coefficient-activity identity and the
careful negative-result accounting are strong. The main weakness is not prose;
it is that the empirical rungs do not yet test one coherent claim at convincing
scale:

- Acrobot supports the proposed `u_N` sampler, but only with a 640-parameter
  policy and one nested-threshold task family.
- The 1.26M-parameter maze result supports a MaxRL-versus-GRPO coverage ordering,
  not the proposed score.
- Digits rejects the stronger estimator-to-sampler mapping.
- Countdown's historical result is a three-seed aggregate with a nonstandard
  coverage proxy, and the clean E2c replacement has not run.
- The only source-faithful named-method comparison is dominated by a protocol
  that spends about 93% of its budget on probes, so it does not establish
  competitiveness against modern online selectors.

The strongest route is to make this a focused paper about **rollout-aware
coefficient activity as an acquisition score**, finish the neural score test,
and demote failure recycling unless E2c provides clean raw-outcome evidence.
Do not add another broad domain before those two existing experiments resolve.

## Current status snapshot

### Paper and artifact

| Item | Status | Consequence |
|---|---|---|
| Compact ICLR body | Current through the 2026-08-14 posterior/event derivations and artifact P0 correction | The theory paper is defensible, but adversarial review holds competitive-method framing until a direct neural score result and matched PLR/ACCEL evidence exist. |
| Page bound | Two cached Tectonic 0.16.9 builds are byte-identical; the conclusion is on page 8, references begin on page 9, and the compact artifact is 15 pages including references and appendix. An independent build audit reports `P0=P1=P2=0`. | The main-text and deterministic-local-build gates are met. Recheck if any manuscript, figure, build-script, or toolchain input changes. |
| Local/website PDFs | `paper/main_iclr.pdf` and `docs/paper-iclr.pdf` now equal the verified compact build (`36a6c1fb...`); `paper/main.pdf` is the deterministic 25-page extended build (`25023b85...`) | These are the current review artifacts. Exact tool/input/log hashes are in `autoresearch/iterate-260814-0112/PAPER_BUILD_RECEIPT.md`. |
| Claim trace | 62/62 quantitative rows traced: 35 exact, 27 rounded | The checked-in compact registry is truthfully identified as 53 rows; the release-only 562-row registry is a distinct object and is not silently substituted. |
| `reproduce.sh` | Non-build path passes in a disposable copy; the optional raw maze-log check is absent; the build path requires an exact Python/Matplotlib/font environment and pinned cached Tectonic, verifies all regenerated figure bytes and logs, then publishes both compact targets plus the extended PDF with rollback | Two independent figure renders and two independent TeX builds produced identical bytes. Keep both explicit tool paths and the frozen source epoch when rebuilding. |
| Build portability | Exact figure-package/font records plus the Tectonic executable, URL mapping, 483-member bundle tree, index, and format are hash-bound, but the executable/cache assets remain external | Before artifact release, provide a checksum-bound bootstrap/container; the present receipt proves deterministic bytes when those assets are supplied, not a clean-checkout build on an unrelated host. |
| Reproducibility data | Several central raw artifacts remain external or missing | The paper is auditable at summary level, not fully reproducible from this checkout. |

### Experiments

| Study | Status | What it supports |
|---|---|---|
| Exact mass/truncation algebra | Complete; enumeration tests pass | Strongest contribution: `A_N(p)=2(pass@N-pass@1)` under the stated practical convention, plus `T=N-1`. |
| Fixed-completion CPU `N` sweep | Complete, post-guidance | `u_N` beats `u_2` for every tested `N>2` on one co-designed synthetic chain; not a general scaling law. |
| Acrobot V2 score tournament | Complete, 20 paired seeds | Best direct positive evidence: `u_16-u_2=+.0480`, CI `[+.0209,+.0738]`; `u_16` also beats uniform. |
| Paid-probe ProCuRL semantics | Complete, 80 paired seeds | Registered superiority unsupported; probe cost dominates. Useful boundary, weak competitiveness evidence. |
| Digits estimator-by-sampler factorial | Complete, negative | Rejects a universal estimator-to-sampler mapping and shows activity is not a learning theorem. |
| Maze estimator factorial wave 2 | Complete, six fresh seed blocks | MaxRL has higher time-integrated coverage than GRPO in 6/6 blocks under each sampler at common settings. It does not test the score itself. |
| Countdown historical aggregate | Complete but artifact-limited | Motivates reporting mean and standard pass@k; it is not a clean causal recycling result. |
| GATE-DR | Complete, negative on 2026-08-13 | Gate dosage is graded, but no useful operating point reproduced. Close this branch. |
| E2c raw-outcome replay control | Integrity-ready, not launched | At the recorded 2026-08-13 18:52 EDT check, GPU use was 8,865 MiB versus the frozen 4,096 MiB ceiling; a 2026-08-14 read-only check still found 8,858 MiB. B1/B2 seed 3, reservoir, three replay runs, delivery validation, and nine-arm evaluation remain. |
| MAZE-SCORE | DRAFT v2; local protocol/analyzer, CPU/import smokes, and endpoint-blind full-arm cost job 9366552 pass | Highest-value live experiment. Exact `u_32`, phase-separated RNGs, strict timepoints/analyzer, content-addressed engineering bundle, and verified Hopper retrieval are implemented. Evidence remains held pending outcome-blind sample-size choice and a clean FROZEN bundle. |
| AMaze UED benchmark | Source-faithful CPU DR/PLR and Frontier v3 tests pass; Hopper import/JIT job 9366896 and one-update job 9366897 passed; the two-phase terminal chain is independently audited and its fresh exact-bundle import job 9367063 is queued. Tie-aware overlay v4, local sibling bundle `d602ce7854f8...`, bounded remote-contract snapshot `da74eb3e0d...`, and the matched `N={2,4,8}` factorial package pass their scoped local checks. | This is the clean competitive lane for PLR/PAIRED/ACCEL claims, separate from MAZE-SCORE and still without a paper endpoint. Independent audit keeps the v4 remote ladder on HOLD for four primary blockers: protected-overlay compatibility, R2 job identity, system-Python GPU probing, and invalid required MIG-gpumem accounting. V4 and factorial production/endpoints also require a new audited runtime identity and 100-update cost/package rung. |
| Frontier calibration telemetry | Static DRAFT GO (`P0=P1=P2=0`): protocol `4053c520...`, analyzer `19b07d2f...`, 48 tests plus independent clock/overflow/alias/forgery/lifecycle attacks | The specification/analyzer are ready to guide implementation, but no telemetry, Hopper, Cost-100 dependency, endpoint, or paper use is authorized without a separate overlay/writer/driver/campaign and independent runtime audit. |
| ICRA/BARN | Campaign `barn-icra2027-20260814-002` was canceled outcome-blind after an unsupported directory-publication operation; replacement campaign `barn-icra2027-20260814-003` is running the same frozen four cells and five seeds under the amended hard-link publication contract. Task granularity remains scientifically mismatched to exact course-level activity. | No robotics paper evidence has been opened here. Preserve campaign 003 and interpret it as a stratum-priority heuristic unless sealed post-merge course data support homogeneity. |

## P0: make the method-to-code contract unambiguous before new evidence

This is the most urgent finding from this audit.

The paper and Acrobot implementation define

```text
u_N(p) = 1 - (1-p)^N - p.
```

The codebase also retains legacy teachers that use

```text
(1 - (1-p)^N) (1-p)
  = 1 - p - (1-p)^(N+1)
  = u_(N+1)(p).
```

The important code-path distinction is:

- `frontier_rl/teacher.py` is correct.
- `curriculum_maxrl/teachers.py::AdvMassTeacher` is correct.
- `curriculum_maxrl/teachers.py::MaxRLFrontierTeacher` uses the shifted legacy
  formula.
- `curriculum_maxrl/maze_gpu/train.py::FrontierTeacher` (teacher key
  `frontier`) uses the shifted legacy formula.
- `curriculum_maxrl/maze_gpu/train.py::FrontierUNTeacher` (teacher key
  `frontier_un`) computes the paper's exact `u_N` formula.
- `curriculum_maxrl/verl_curriculum.py::FrontierTeacher` uses the shifted formula.
- `hopper/sbatch/maze_score_array.sbatch` selects `frontier_un`, so the planned
  MAZE-SCORE arm with `--rollouts 32` does run exact `u_32`, consistent with
  `hopper/MAZE_SCORE_PREREG.md`.

At `N=32`, the legacy formula's peak would move only from about `.105775`
(`u_32`) to `.103508` (`u_33`). This does not invalidate the selected
MAZE-SCORE arm. The coexistence of two easily confused teacher names is still a
submission risk because the paper's central promise is an exact mapping from
the deployed estimator to the deployed sampler.

Required actions before MAZE-SCORE freezes:

1. Preserve the verified `frontier_un` selection in the frozen source bundle;
   do not switch it to the legacy `frontier` key. A smoke result remains
   engineering-only.
2. Put the exact formula in one canonical, tested function and import it from
   every active teacher. Do not silently rewrite historical source locks.
3. Add tests for `N=2 -> p(1-p)`, vector/scalar parity, exact equality across
   active adapters, and the mapping from configured rollout count to exponent.
4. Exercise the actual `frontier_un` dispatch in a test or smoke, record source
   hashes, and freeze the preregistration and source together. If the protocol
   has already become frozen by the time this is acted on, add only a dated,
   outcome-blind clarification.
5. Classify past maze runs by their recorded teacher key: `frontier_un` is the
   exact deployed-`N` score, while `frontier` is the legacy shifted heuristic.
   Do not infer the implementation from a generic "frontier" label alone.
6. Deprecate the ambiguous `MaxRLFrontierTeacher` name and retain one public
   name for the paper method, such as `CoefficientActivityTeacher`.

## P0: repair the evidence-execution contract before MAZE-SCORE

The first live Hopper audit found that the formula is not the launch blocker;
the execution contract is. Job 9361275 was canceled while still pending because
it would have consumed a 3g.40gb allocation and then failed or used stale state:

- the sbatch copied only `maze_gpu/*.py`, but `train.py` imports the parent
  `curriculum_maxrl/estimators.py`;
- `cp -n ... || true` silently accepted missing copies and reused old files;
- the Slurm stdout directory was created inside the job, after Slurm needs to
  open it;
- one fixed work directory and seed-only checkpoint names could mix smoke,
  retries, and evidence from different source revisions;
- the promised MAZE-SCORE analyzer did not exist.

There are also two outcome-blind scientific protocol defects:

1. The historical loop evaluates after updates 1, 26, ..., 226, and 250, then
   evaluates the unchanged final model again. The reused reader averages all
   nonnegative records, so it includes the first-update record and double
   weights the endpoint rather than the stated ten checkpoints 25--250.
2. The first arm creates SFT and consumes Python/NumPy/Torch randomness; later
   arms load the checkpoint without consuming it. Thus the claimed common
   post-SFT random stream is false and retry behavior depends on whether a
   checkpoint already exists. Evaluation sampling also shares the mutable
   training RNG, and logging `teacher.distribution()` advances Thompson state.

Required launch gates:

1. Preserve the historical behavior under an explicit legacy protocol; use a
   separately named MAZE-SCORE protocol whose completed-update checkpoints are
   exactly `{25,50,...,250}`, each once.
2. Use independent deterministic SFT, RL-rollout, teacher, task, and evaluation
   streams. Reset the post-SFT streams identically whether the warmstart was
   created or loaded, and make logging observational rather than state-changing.
3. Pre-generate or content-address each warmstart, record its hash, and require
   the three arms within a block to use the same hash.
4. Stage a complete immutable source bundle with a SHA-256 manifest. Every
   campaign/attempt gets new source, log, result, telemetry, and metadata paths;
   partial attempts are retained and never overwritten or silently resumed.
5. Freeze and test the analyzer before evidence. It must reject missing,
   duplicate, or extra checkpoint rows and any mismatch in campaign, source,
   score family, rollout count, effective exponent, or warmstart hash.
6. Define the inference completely: paired percentile bootstrap interval,
   exact paired sign-flip test on the mean, Holm family and adjusted decision,
   timeout/NaN/incomplete-block handling, and an outcome-blind retry rule.
7. Use three conclusions: `supported`; `practically ruled out` only if the
   interval's upper bound is below the +.005 SESOI; otherwise `inconclusive`.
   Failure to reject is not evidence that transfer failed.
8. Pass a CPU submit/write/fetch/checksum smoke, then an import-only 1g.10gb
   GPU smoke, then one complete non-evidence arm before freezing the core array.

Implementation snapshot on 2026-08-14: gates 1--7 are implemented and covered
by 20 focused tests; CPU job 9366532 and import-only GPU job 9366547 completed,
were retrieved, and passed hash verification. The engineering source bundle is
content-addressed as `f4359095fb05490192b4`. Full-arm cost job 9366552 also
completed and passed an endpoint-blind manifest/schema audit in 22:22, with
peak GPU memory 39,672 MiB on `3g.40gb`; retain that slice. Gate 8 is therefore
complete. The core array is still fail-closed on a DRAFT/dirty bundle and has
not been submitted.

Before freezing the array, make one explicit outcome-blind power/budget choice.
Using SESOI `.005` and the pessimistic historical paired SD `.0135`, the
candidate 30-block design has only about 50.1% power at alpha `.05` and 38.2%
at the first Holm threshold `.025`. Sixty balanced blocks reach about 80.6%
unadjusted but 71.4% at `.025`; 72 balanced blocks reach about 87.3% and 80.1%,
respectively. For the strongest paper, prefer 72 blocks (candidate seeds
20--91) if quota permits; otherwise label 60 or 30 explicitly as a budget/
precision compromise. Conservative three-arm upper bounds are 33.55, 67.10,
and 80.52 3g.40gb MIG-hours for 30, 60, and 72 blocks before shared-warmstart
savings. Any change from the tested 8 CPU/60G contract should be separately
re-smoked; CPU/RAM were overrequested, but GPU memory was not.

## Recommended paper thesis and scope

Use this central claim:

> For a specified group estimator and rollout budget, coefficient activity is
> an exactly computable, rollout-aware acquisition signal. It does not guarantee
> learning progress, but it yields a testable task-selection rule whose value and
> boundary can be measured under controlled budgets.

This is stronger than the current broad “curricula and failure recycling depend
on the estimator” framing because every main contribution can be made to answer
one question: **does the estimator-derived activity profile help select tasks?**

Recommended contribution ladder:

1. **Theory:** exact practical-MaxRL coefficient activity, peak, finite-`N`
   behavior, `T=N-1` convention correction, and the dependent-rollout identity.
2. **Method:** one canonical coefficient-activity teacher with explicit
   posterior, exploration, concentration, and cost contract.
3. **Evidence:** direct score-shape tests from controlled small scale to neural
   scale, including a strong negative counter-test and competitive baselines.
4. **Measurement rule:** retain raw task outcomes and report mean@k with standard
   pass@k. This can be a recommendation, not a separate recycling contribution.

Failure recycling should remain in the abstract and contribution list only if
E2c produces a delivery-valid three-seed result with retained outcomes. If E2c
is blocked, fails delivery, or is heterogeneous, move the historical Countdown
aggregate and gate studies to a short diagnostic subsection or appendix. The
current aggregate is too weak to carry one third of the paper.

## Experiment program, in priority order

### P1. Finish MAZE-SCORE after the P0 repair

This is the best existing experiment because it closes the exact gap between the
Acrobot score result and the neural maze result.

Keep the scientific contrast while replacing the defective execution details:

- fixed MaxRL estimator and `N=32`;
- exact `u_32`, `u_2=p(1-p)`, and uniform arms;
- fresh paired seed/warm-start blocks, with the final count fixed from an
  outcome-blind power/precision calculation before launch;
- time-integrated held-out coverage primary;
- paired seed-block inference, declared SESOI, Holm control for the two tests;
- no endpoint-log or result inspection before the complete matrix.

Ten blocks are too weak for a confident +.005 claim. The completed full-arm
profile and conservative historical paired SD `.0135` imply that 30 blocks
are substantially underpowered, 60 reach about 71% power at the first Holm
threshold `.025`, and 72 reach about 80%. Freeze 72 balanced blocks if quota
permits; otherwise describe 60 or 30 explicitly as a precision-versus-cost
compromise, without consulting new endpoint values.

Use a fresh primary evaluation panel, keyed independently within each seed
block and shared across its arms. The repeatedly used seed-12345 panel can be a
descriptive continuity check, not the new confirmatory primary. Increasing the
number of mazes per level is often cheaper and more valuable than adding noisy
checkpoint evaluations; choose both before the smoke receipt freezes.

Add only outcome-blind logging, without changing the primary:

- configured score exponent and source hash;
- sampled-task distribution and its entropy/effective support;
- posterior mean and calibration by difficulty level;
- realized group `K`, exact realized coefficient mass, silent-group rate, token
  or environment-step cost, and optimizer updates;
- checkpoint-level mean@8 and standard observed-set pass@8 from raw outcomes.

These diagnostics test whether the intervention was delivered and whether the
score increased its proximal target. Treat any learning-versus-mass mediation as
descriptive unless it is separately preregistered.

Decision branches:

- **Supported:** make MAZE-SCORE the primary empirical figure; present Acrobot as
  the controlled replication and Digits as the boundary.
- **Practically ruled out:** only when the interval upper bound lies below the
  SESOI; report it at equal prominence and narrow the method claim.
- **Inconclusive:** retain the calibrated small-scale claim, show the interval,
  and do not phrase non-significance as failed transfer.

### P1b. Add a source-faithful AMaze UED lane

The new competitive navigation lane should use
`facebookresearch/minimax` v0.2.0 at immutable commit
`d053054c5290a04c1c4cd8b55704d999cad73e30` (Apache-2.0). Do not use the
archived original PLR repository as the executable substrate: it is
CC-BY-NC-4.0, predates the final paper, lacks the published unbounded-buffer
implementation, and its README defaults do not reproduce the paper's replay
schedule or 100-episode final evaluation.

For audit-only reproduction of that archived code, record three divergences
explicitly: its default `level_replay_rho=1.0` waits until the finite level set
has been seen, whereas the paper-style annealed replay schedule requires
`rho=0.0`; its final evaluator defaults to 1,000 rather than the paper's 100
episodes; and the appendix's full-distribution/top-M variant is not implemented
(`full_train_distribution` disables the sampler). These are reasons to run
direct minimax baselines under one frozen evaluator instead of mixing numbers
from incompatible PLR scripts ([PLR paper](https://proceedings.mlr.press/v139/jiang21b.html),
[archived PLR code](https://github.com/facebookresearch/level-replay)).

The first intervention must be score-only. Keep the official robust-PLR
student, PPO, generated levels, buffer, replay probability, inverse-rank
transform, staleness mixture, architecture, and evaluator. Replace MaxMC only
with a per-level online Beta posterior over sparse-goal rollout success and

```text
u_N(p) = 1 - (1-p)^N - p.
```

There are two scientifically different variants and they must not share a
label. The exact grouped-MaxRL lane uses `n_parallel=4`, `n_eval=8`, and `N=8`:
four task instances are each repeated across eight rollout streams, preserving
the official total of 32 simultaneous rollouts and the same student-transition
budget. Each stream contributes one Bernoulli observation (`any` goal before
the rollout ends), even if an early success auto-resets and completes another
episode; otherwise the effective `N` would vary by task. A second low-cost
bridge may accumulate a posterior across ordinary visits while
keeping the official `n_parallel=32,n_eval=1` setting. In that bridge, `N>1` is
a counterfactual/deployment-aware priority, not the exact activity of the
current PPO batch; call it `Posterior-Frontier-PLR` and report the mismatch.

The exact 4-by-8 comparison must also match replay onset rather than blindly
copying the official 4,000-level buffer. At `min_fill_ratio=.5`, the official
32-by-1 arm reaches 2,000 filled slots after about 63 all-new cycles, or roughly
512k evaluation-stream transitions. A 4-by-8 arm with the same 4,000 slots
would need about 500 cycles and 4.096M transitions before robust-PLR can train:
an eightfold warm-up confound. Therefore the primary score-isolation pair is
coefficient activity versus upstream MaxMC with identical
`n_parallel=4,n_eval=8,buffer_size=500,min_fill_ratio=.5`. Keep the official
MaxMC 32-by-1/4,000 arm as the source-faithful external reference and the
Frontier 32-by-1/4,000 bridge as a compute-shaped ablation. If a 4-by-8/4,000
arm is retained, label it a capacity sensitivity and compare it at fixed
environment-step checkpoints; do not use it as the sole primary control.

The highest-value mechanistic follow-up is a small matched group-size
factorial, frozen before opening the N=8 development endpoint. Use
`N in {2,4,8}`, `n_eval=N`, `n_parallel=32/N`, and
`buffer_size=4000/N`, with coefficient activity and MaxMC paired within every
layout. Every cell then runs 32 streams per outer cycle and reaches 50% buffer
fill after about 63 all-new cycles, so replay onset costs the same roughly
512k student transitions. The 63-cycle equality assumes every proposed new
group inserts a distinct level; accepted receipts must report zero
duplicate-new groups or else use the observed fill cycle/transition count
rather than the nominal value. The number of distinct levels and eventual buffer
capacity still change with N, but the within-N Frontier-minus-MaxMC contrast
removes that layout effect; the contrast-of-contrasts tests whether deployed
rollout count changes the value of the score itself. Treat training seed as
the unit, retain fixed-transition and fixed-update readouts, and report task
exposure/effective support. This factorial is stronger evidence for the
rollout-aware mechanism than tuning several unrelated posterior priors. With
five development seeds its interaction is descriptive (the minimum nonzero
two-sided sign-flip p-value is `.0625`); a paper claim must either rerun the
selected N on untouched confirmatory seeds or freeze the full interaction
family and its multiplicity correction before those endpoints exist.

Freeze `N`, prior alpha/beta, actual `n_eval`, and posterior treatment. For a
Beta(`a`,`b`) posterior, the principled deterministic candidate is the analytic
posterior expectation

```text
E[u_N(p)] = 1 - (b)_N/(a+b)_N - a/(a+b),
```

not the plug-in `u_N(a/(a+b))`. Compare those two on development seeds because
the plug-in systematically overstates expected activity for `N>=2`; use the
analytic expectation as the theory-preferred candidate. Thompson sampling is a
later, separately labelled exploration ablation. Code and analyzer receipts
must log posterior mode, `N`, and `n_eval`, with a strict option that rejects a
group-size mismatch.

Make the frontier interpretation explicit in the paper rather than presenting
the score as a heuristic. For `N>1`,

```text
u'_N(p) = N(1-p)^(N-1) - 1,
u''_N(p) = -N(N-1)(1-p)^(N-2) < 0,
p*_N = 1 - N^(-1/(N-1)).
```

Thus the score has a unique preferred success rate fixed by the attempted-
rollout budget: `p*_2=.5`, while the exact `N=8` arm targets `p*=.257`. This is
a testable distinction from generic learnability and explains why increasing
`N` moves the curriculum toward harder solvable levels. Because `u_N` is
strictly concave, Jensen's inequality gives
`E[u_N(p)] <= u_N(E[p])`; the analytic Beta score's gap from the plug-in score
is therefore a principled epistemic-uncertainty penalty. Report posterior
concentration and staleness-selection rates so that this conservatism is not
mistaken for lack of exploration.

The cleanest operational interpretation is
`u_N(p)=P(r_1=0, at least one of the other N-1 rollouts succeeds)`. It is the
probability that a designated rollout fails while its purchased peer group
contains positive evidence. This exposes both zero-activity regimes and gives
reviewers a concrete reason that deployment group size belongs in the score.

Turn that identity into a falsifiable mechanism test. Before incorporating a
new group into its posterior, record the level's predicted analytic activity
and the realized half-mass
`m(K)=1{0<K<N}(N-K)/N`. Under the stated Bernoulli model,
`E[m(K)|history]=E[u_N(p)|history]`. Report calibration by frozen prediction
bins, mean absolute calibration error, and squared prediction error. MaxMC is
not a probability forecast: retain only its stored **pre-group** replay score
for discrimination diagnostics, and compare the activity actually purchased
by each sampler without calling MaxMC calibrated. These
diagnostics directly test whether Frontier forecasts the estimator-side
quantity it claims to target; they are stronger than explaining a solved-rate
gain post hoc. All predictions must be pre-update to prevent using the current
group twice. Treat training seed—not purchased group—as the independent unit,
and reconcile robust-PLR outer cycles separately from PPO updates because
exploratory cycles purchase groups without applying an optimizer update.

The same mechanism also has a closed-form predictive noise floor, which should
be frozen before interpreting calibration error. For
`m_N(K)=1{0<K<N}(N-K)/N` and `K|p ~ Binomial(N,p)`, exact enumeration gives

```text
E[m_N(K)^2 | p]
  = (1-p)^2 + p(1-p)/N - (1-p)^N.
```

Under `p|history ~ Beta(a,b)`, with `s=a+b`, this becomes

```text
E[m_N(K)^2 | history]
  = b(b+1)/(s(s+1)) + ab/(N s(s+1)) - (b)_N/(s)_N,
Var[m_N(K) | history] = E[m_N(K)^2 | history] - q^2,
```

where `q=E[u_N(p)|history]`. Independent enumeration over
`N={2,3,4,8,16,32}` and boundary/interior probabilities verifies the first
identity, and positive posterior-predictive variances were checked for the
planned prior families. Add this as a secondary standardized-residual and
noise-floor diagnostic after the telemetry clock contract is audited; it does
not turn adaptively purchased groups into independent inferential replicates.

This is not merely a scalar calibration: heterogeneous visit counts can make
the analytic and plug-in priorities reverse level rankings. In every
development run, log both scores for every filled buffer slot, their rank
correlation and top-k overlap, the Jensen gap, and replay probability mass by
posterior-trial-count quartile. These are outcome-blind delivery diagnostics.
They reveal whether any performance difference comes from the intended
uncertainty penalty or simply from selecting older, better-measured levels.

Before any development endpoint, repair or explicitly ablate rank ties. The
pinned PLR transform performs a stable sort and assigns weights proportional
to `rank^(-1/.3)`; it does not give equal scores equal probability. At
temperature `.3`, rank 1 receives about 87.16% of the score-only distribution
and the top five receive 99.31%. After a new N=8 group, the Beta(1,1)
coefficient score has only nine possible values (one for each success count),
so stable slot order can dominate replay among large tied blocks. This is much
more severe for a binary posterior score than for path-length-sensitive MaxMC.
Add a deterministic tie-aware transform that gives every member of a tie block
the same share while preserving that block's total upstream rank mass, and
apply it to both Frontier and the group-matched MaxMC control. Retain the
unaltered upstream MaxMC arm as the source-faithful reference. Tests must cover
permutation invariance within ties, total mass preservation, all-equal scores,
unfilled slots, and mathematically unchanged behavior when scores are distinct.
Because upstream normalizes in buffer-slot order, exact normalized bit parity
for every distinct-score permutation is incompatible with exact permutation
equivariance in float32; freeze a tight rounding tolerance for that comparison
while keeping the opt-out/default path bit-identical. Log tie-block
sizes and effective support. The current v1 terminal smoke may validate I/O and
cost, but do not launch its five-seed performance gate until this issue is
resolved in a newly frozen protocol/overlay.

Concretely, if an exact-score tie occupies one-indexed sorted ranks `l..r`,
assign each member the same pre-normalization weight
`mean_{j=l..r} j^(-1/beta)`. This preserves exactly the mass that upstream
rank prioritization assigned to those ranks, is permutation invariant inside
the tie, and reduces to upstream behavior when `l=r`; the existing staleness
mixture is then applied unchanged. Do not use epsilon jitter, which merely
relabels the arbitrary order and makes the result scale-dependent.

The option named `force_unique` does not make the sampled PPO task groups
unique. Replay indices are drawn sequentially **with replacement**; the later
deduplication pass only controls buffer updates (and, for Frontier, deliberately
retains repeated observations of an already-buffered level). Thus the frozen
4-by-8 lane can train all four task groups on the same replay level. With the
score-only rank distribution at `temp=.3` and a 500-slot buffer, four draws
have only `1.468` distinct levels in expectation and select rank 1 four times
with probability `0.577` before the 0.3 staleness mixture is applied. This is
not an implementation error by itself, but it makes arbitrary tie breaking a
task-diversity confound. The tie-aware implementation and run receipts must
therefore also report sampled distinct-level count, duplicate-group count, and
replay effective sample size. Do not claim that `force_unique=true` removes
this issue. A later without-replacement sampler may be tested as a matched
ablation, but changing sampling and scoring together would destroy the primary
score-only comparison.

An outcome-blind prior-predictive stress case quantifies the confound. The
4-by-8/B500 arm first crosses its replay gate with 252 filled levels. Under a
Beta(1,1) prior, one eight-trial group's prior-predictive success count is
uniform over `K=0..8`; assigning 28 slots to each count therefore gives nine
equal-size exact-score blocks. The maximum-score (`K=2`) block receives
`.999850` of the score-only mass under both transforms, but stable ranking gives
its first slot `.871570` and has score-distribution ESS `1.303`. The
block-mass-preserving transform gives each of the 28 top-tied slots `.035709`,
ESS `28.008`, and raises expected unique levels in four draws from `1.468` to
`3.791` while leaving the block's total priority unchanged. This is an
illustrative prior-predictive construction, not a claim about realized buffer
composition; it belongs in the method sanity checks rather than the result
table.

This repair is now implemented and independently audited for bounded
engineering as overlay v4. Its contract is
`3d5f3827a82a4f713314091289196a1c2909dd5d7c4c96dd532052c5706e832b`
and its applied-manifest SHA-256 is
`9b411f61ebc56bb93fc22cad6b19299c38eab2b696fa17f7783c7729e1db02ae`.
The implementation uses deterministic segmented scans and canonical tie-mode
normalization, contains no repeated-index scatter reduction, and makes actual
score and replay probabilities exactly permutation-equivariant. Independent
JAX 0.4.31 checks passed 34 core/legacy tests, one bounded one-update E2E, six
terminal/two-arm tests, 400 hostile tied-buffer permutations, 5,000 distinct
permutations, and default/sampling parity. The largest measured normalized
distinct-score rounding difference from source stable ranks was `3.5763e-7`,
below the frozen `5e-7` bound; opt-out behavior remained exact. The original v1
protocol and v3 overlay remain byte-bound history, while the tie-aware protocol
is a separate DRAFT v2 with `production_driver_authorized=false`. Therefore
this is a method-delivery pass, not authorization for a development or paper
endpoint. Before a general-purpose release, also harden the direct constructor
to reject non-finite temperature; current authored configs/drivers already bind
finite `temp=.3`, so this P2 cannot affect the frozen engineering receipts.

The rollout-count mechanism is now specified as a separate, outcome-blind
factorial DRAFT rather than an informal follow-up. It contains six matched
cells: Frontier and MaxMC at `N={2,4,8}`, with `n_eval=N`,
`n_parallel={16,8,4}`, and `buffer_size={2000,1000,500}`. Every cell therefore
processes 32 streams and 8,192 student transitions per outer cycle. Within each
`N`, the arm pair differs only in score and Frontier-only posterior fields, so
that contrast isolates score choice conditional on the shared layout. Across
`N`, rollout grouping, level diversity, and buffer size change jointly;
neither cell means nor contrast-of-contrasts identify a pure estimator-`N`
effect.

The DRAFT protocol SHA-256 is
`81a57668d3cfdf595f13710df6152a437b8c4640791fbeeed2ef8c9e9486f26f`
and its six-config closure manifest is
`58e1ffd9c7e3d80992971b331c540d6c8976c9cd4082391fae92de0df4fd417f`.
Independent pinned-JAX validation passed 12/12 tests, parsed all six configs,
generated six unique run identities, and rejected direct-parser `xpid=latest`,
cross-`N` checkpoint reuse, and strict `N != n_eval`. The contract also binds
matched within-`N` update/gradient/optimizer/cycle/transition counts and a new,
round-tripped post-loop checkpoint; a periodic checkpoint is not an admissible
endpoint or recovery source.

The nominal replay warm-up equality is conditional, not automatic. If every
fill cycle accepts exactly `n_parallel` distinct new groups, all three layouts
cross half-buffer fill after 63 cycles, or 516,096 transitions. Any duplicate,
partial, nonfinite, or rejected new group invalidates that label; the receipt
must instead report the observed fill trajectory and replay-eligibility cycle.
The package is explicitly `DRAFT_ENGINEERING_ONLY_NOT_PRODUCTION_AUTHORIZED`,
and no performance, OOD, Hopper, or GPU endpoint was accessed during its
construction or audit.

The implementation must fail closed on delivery drift. In strict mode every
real level cell must contribute exactly `n_eval` completed streams; partial
cells are rejected and counted. Repeated samples of an existing replay-buffer
level must accumulate all Bernoulli evidence, while duplicate newly generated
levels must be rejected or canonically aggregated rather than silently losing
later groups. Frontier resumes must require the posterior state and an exact
static curriculum signature (score family, `N`, `n_eval`, priors, posterior
mode, threshold, buffer size, and overlay contract); source-faithful checkpoints
without these fields are initialization artifacts, not valid Frontier resumes.
Until multi-device buffer restoration is repaired and tested, fail closed on
`n_devices != 1`.

Do not use upstream `ExperimentRunner.train` as the evidence driver. It does
not return terminal state, and its checkpoint cadence is keyed to outer cycles,
so the last `checkpoint.pkl` can precede the nominal terminal student PPO
update.
Before any 100-update or full run, add a content-addressed driver that owns the
loop, reconciles outer cycles, student transitions, student PPO updates, and
optimizer gradient applications, and atomically writes a terminal checkpoint
plus completion receipt.
Run evaluation through a separately hashed, deterministic external evaluator;
the analyzer must reject periodic/preterminal checkpoints and any missing
terminal counters.

Here `minimax` counter names require care. `n_updates` is a student PPO update,
while the upstream `n_grad_updates` assignment mirrors that counter rather than
counting minibatch gradient applications. With five PPO epochs and one
minibatch, 30,000 PPO updates imply 150,000 optimizer-step applications. The
terminal receipt must report and reconcile all three quantities explicitly.

An outcome-blind matched-development protocol and analyzer now freeze seeds
101--105, the three validation mazes, paired seed 100000+s evaluation RNG,
30,000 PPO updates, 150,000 optimizer applications, terminal receipts, and the
positive/no-environment-worse-than-.05 keep rule. The loop-owning driver saves
the true post-loop checkpoint, reconciles update/optimizer/transition budgets,
and emits a safe per-slot snapshot without analyzer-side checkpoint
unpickling. The schema-2 assembler preserves and revalidates both source
closures, receipts, all 30 ordered raw episode rows, terminal logs/accounting,
the run context, and the Frontier snapshot in one atomic package.

Trainer and evaluator now fail before importing the benchmark unless the
campaign digest, protocol, analyzer, assembler, both drivers,
source/environment provenance, and selected Slurm submission all match. The
evaluator also rejects a wrong execution lane or a trainer protocol/source
receipt that differs from its own. Thirty-one bounded
driver/package/analyzer tests pass. Campaign launch remains `HOLD` until the
same two-phase terminal chain passes on Slurm: Phase A must exit after real
training and external evaluation, while Phase B may add authoritative terminal
`sacct`, logs, and the final package only after `COMPLETED 0:0`. The primary
analyzer does not infer diagnostic ranks from aggregate logs or unpickle model
checkpoints.

Canonical LSTM configuration:

- 30,000 student PPO updates and 32 student rollout streams (32 distinct levels
  in the official arm, or four levels repeated eight times in the exact grouped
  arm), 256 rollout steps, five PPO epochs, one minibatch, LSTM-256, 13x13
  interior, 60 wall draws with replacement, and a 250-step training episode
  horizon. The three fixed singleton validation mazes each resolve to a
  distinct 450-step evaluation horizon contract;
  robust PLR may require more than 30,000 outer cycles because exploratory
  new-level cycles skip the PPO update;
- validation only on `SixteenRooms`, `Labyrinth`, and `StandardMaze` while
  developing; never select variants on the full test panel;
- final frozen twelve-environment panel:
  `SixteenRooms`, `SixteenRooms2`, `Labyrinth`, `Labyrinth2`, `StandardMaze`,
  `StandardMaze2`, `StandardMaze3`, `Crossing`, `FourRooms`, `SmallCorridor`,
  `LargeCorridor`, and `PerfectMazeMedium`;
- source-derived maximum horizons for that exact order are
  `[450,450,450,450,450,450,450,484,250,450,882,882]`. Thus ten episodes
  per environment cost at most 60,980 external-evaluation transitions per
  checkpoint. Freeze and runtime-verify the full vector; do not project the
  three-validation-maze 450-step constant onto the larger panel;
- ten stochastic-policy episodes per environment and one shared evaluation seed
  per paired training block. Ten confirmatory training seeds are a floor, not
  a fixed sample-size justification; freeze the final count from a declared
  absolute solved-rate SESOI, a conservative paired-SD bound, and the exact
  multiplicity family after development but before confirmatory endpoints;
- primary metric: mean solved rate over the twelve environments, followed by
  the paired training-seed contrast and interval. Preserve per-environment
  solved rates so improvement on easy rooms cannot hide regression on mazes.

For the confirmatory headline, define one number per paired training seed:
the Frontier-minus-ACCEL difference after averaging the exact same twelve
environments and frozen evaluation episodes. The training seed, not an episode
or maze, is the inferential unit. Declare superiority only if the two-sided 95%
paired confidence interval excludes zero in Frontier's favor; separately report
whether the point estimate clears the predeclared `.02` SESOI. Use a paired
t interval/test as the primary analysis once the planned sample-size floor is
met, with an exact sign-flip test and seed-level bootstrap interval as frozen
sensitivity analyses. Do not multiply 12 mazes by 10 episodes and call those
120 independent replicates. Report the full seed-by-maze matrix and episode
counts so binomial evaluation noise remains auditable.

The [published minimax 30k-update LSTM
references](https://arxiv.org/html/2311.12716v3) are DR `.55±.05`, PAIRED
`.63±.04`, robust PLR `.82±.02`, parallel PLR `.80±.02`, ACCEL `.83±.02`, and
parallel ACCEL `.78±.03` (means and standard deviations over ten runs). A claim
of beating the strongest published baseline requires a direct matched ACCEL
rerun and a favorable paired interval, not merely a point estimate above
`.83`.

Keep three benchmark questions separate. The pinned upstream configurations
make them different experiments, not interchangeable labels:

| Question | Required arm | Frozen geometry / selector controls | Permitted claim |
|---|---|---|---|
| Did the new priority help under exact grouped evidence? | Tie-aware Frontier versus tie-aware MaxMC | Both `4 x 8`, buffer 500, replay `.5`, min-fill `.5`, staleness `.3`, rank temperature `.3`, robust sequential PLR, 32 streams and 8,192 training transitions per outer cycle | Score effect conditional on the grouped layout |
| Is the result competitive with released robust PLR? | Pristine upstream MaxMC reference | `32 x 1`, buffer 4,000, replay `.5`, min-fill `.5`, staleness `.3`, rank temperature `.3`, robust sequential PLR, with no overlay-only arguments | Direct matched rerun of the released PLR configuration |
| Does Frontier improve the strongest released generator? | Posterior-Frontier-ACCEL versus pristine ACCEL | Both `32 x 1`, buffer 4,000, replay `.8`, min-fill `.5`, staleness `.5`, temperature `.3`, 20 default mutations, batch mutation criterion, subsample 4, and identical PPO/generator settings | Score/parent-priority effect inside ACCEL; not exact current-batch `N=8` activity |

The official parallel variants are separate throughput algorithms: parallel
PLR and parallel ACCEL set parallel evaluation and change staleness and, for
parallel ACCEL, the learning rate and entropy coefficient. Include them only
when making a wall-time or parallelism claim. PAIRED is a two-student learned
teacher with a different runner and budget decomposition; it belongs in the
final competitive table but is not a score-isolation control. Every generated
command must be diffed against the pinned JSON before launch, and the receipt
must state whether the arm is `source_faithful_reference`,
`geometry_matched_score_test`, or `generator_matched_extension`.

Do not inherit the published ten-run count mechanically. For scale, a
two-sided normal approximation with SESOI `.02` and paired SD `.03` needs about
18 pairs for 80% power at alpha `.05`; SD `.04` needs about 32, before a
small-sample or multiplicity allowance. These are planning examples, not a
frozen power result. Predeclare one primary Frontier-versus-ACCEL contrast if
that is the headline; otherwise account for the full comparison family. Use a
blinded or conservative variance rule and cap compute outcome-blindly. If quota
forces ten pairs, report the resulting interval and call the design a
precision/compute compromise rather than “well powered.”

Budget reporting is unusually important here. A target of 30,000 student PPO
updates does not imply equal exposure: DR uses 245.76M student transitions,
PAIRED has two students, robust PLR needs additional outer cycles because it
skips PPO updates on exploratory rollouts, and parallel PLR/ACCEL double or
triple rollout batches. Record and compare:

1. outer cycles;
2. student, teacher, replay, and mutation transitions separately;
3. actual student PPO updates;
4. optimizer gradient applications (`PPO updates × epochs × minibatches`);
5. evaluation transitions excluded from training budgets;
6. GPU model/slice, peak memory, wall time, and energy if available.

The evaluation drivers originally understated this budget by treating the
fixed singleton mazes as 250-step environments. In the pinned code,
`SixteenRooms`, `Labyrinth`, and `StandardMaze` each resolve to
`2*(13+2)*(13+2)=450` steps. The corrected external evaluation cost is therefore
`3*10*450 = 13,500` transitions per terminal checkpoint, and each periodic
evaluation call has the same maximum. Both drivers now derive these horizons
from the constructed environments, fail on drift, and pass the full 31-test
terminal/package/analyzer suite. This correction changes resource accounting, not an
observed performance endpoint; no real endpoint was opened.

Minimum ladder:

1. pinned CPU imports, config generation, repository tests, and one-update DR
   and robust-PLR state/checkpoint smokes;
2. import/JIT-only GPU smoke on the RTX 5090 when uncontended and on Hopper
   A100 with the same source/environment hashes;
3. one-update official-shape DR, PAIRED, robust PLR, ACCEL, and Frontier-PLR
   smokes, followed by a 100-update evaluator/receipt test;
4. fixed development seeds comparing official robust PLR
   (32-by-1/4,000), group-matched MaxMC (4-by-8/500), exact grouped coefficient
   activity (4-by-8/500), the 32-by-1 posterior bridge, and a predeclared `u_2`
   diagnostic; keep/discard variants only on the three validation mazes and
   report both fixed-transition and fixed-update views;
5. one complete DR and ACCEL cost/reproduction run before freezing power,
   schedule, retry, and interaction-matched sensitivity decisions;
6. frozen paired-seed matrix with DR, PAIRED, robust PLR, ACCEL, and one selected
   Frontier-PLR method; ten seeds is the floor and the final count follows the
   predeclared SESOI/variance/multiplicity calculation. Add parallel variants
   only for a speed claim.

Use a staged SOTA decision rather than tuning a large family against the test
panel. First require exact 4-by-8 Frontier to beat its group-matched MaxMC
control on the three validation mazes at matched transitions and updates. Then
require the 32-by-1 posterior bridge to be competitive with official robust
PLR; this diagnoses whether retaining 32 distinct levels offsets the bridge's
counterfactual `N=8` interpretation. Only if one of those gates is positive,
add one predeclared `Posterior-Frontier-ACCEL` arm by replacing ACCEL's MaxMC
replay/parent priority with the same posterior score while preserving its
official 32-by-1 mutation, replay, buffer, and PPO settings. This is the
highest-probability route to a benchmark-leading result because it preserves
ACCEL's level diversity and generator while testing whether Frontier improves
the shared prioritizer. Label it a deployment-aware posterior score, not exact
current-batch coefficient activity. Do not add grouped ACCEL unless mutation
identity and eight-stream evidence aggregation receive their own tests.

In the released sequential ACCEL configuration, mutation criterion is
`batch`: the score does not choose a different mutation operator within the
sampled parent batch. The score-only Frontier intervention must therefore
change only replay/parent sampling, child insertion or eviction, and later
replay priority; keep replay probability `.8`, staleness `.5`, temperature
`.3`, buffer 4,000, 20 default mutations, PPO, and all generator settings
identical. A mutated child starts from its own Beta prior plus its own rollout
observation; do not inherit the parent's success counts in the primary arm.
Parent-to-child posterior shrinkage is a plausible later method extension, but
it changes the causal question and must be separately named and preregistered.

Freeze a practical development keep rule before those runs: advance a variant
only if its paired mean validation solved-rate contrast is positive, no one of
the three validation environments regresses by more than `.05`, its zero-
delivery counters remain exact, and its transition/update accounting matches.
The final “beat ACCEL” statement requires the full twelve-environment panel,
the direct matched adequately sized ACCEL rerun, a positive paired interval, and a
point estimate above the matched ACCEL arm—not comparison only to the published
`.83` reference.

Engineering status on 2026-08-14: source-faithful Python 3.10/JAX 0.4.31
environments were built under `/data` and on Hopper; compatibility checks, a
DR rollout/PPO/checkpoint smoke, and a robust-PLR buffer-warmup/replay/PPO/
checkpoint smoke pass. Frozen Frontier contract
`5868d346ba00e43225424f053ded3dea056b8f76ce4bfdaa1a107a4955cb9000`
passes 20 adversarial and parsed-configuration tests. Exact grouped Frontier
and the 32-by-1 bridge each reach three full-horizon PPO updates locally, and
the exact checkpoint resumes to update four with continuous posterior counts.
The group-matched MaxMC control differs only in the intended score fields.
Two additional batching-contract tests verify that each eight-stream group
shares one task reset while all 32 streams receive distinct environment-step
keys and non-broadcast policy samples. This establishes RNG separation, not
statistical independence of realized outcomes.

A content-addressable terminal training driver and external evaluator now own
the otherwise incomplete upstream endpoint contract. In bounded CPU tests,
both Frontier and group-matched MaxMC produce a true post-loop checkpoint,
four-file analyzer-compatible training directory, separately closed diagnostic
sidecar, and 30 raw evaluation episodes. The evaluator agrees with an
independent upstream `EvalRunner.run` call across all six aggregate fields
within `2e-6`; the Frontier snapshot recomputes both posterior scores from the
same counts and matches the pinned replay distribution within `2e-6`. The full
31-test train/evaluate/assemble/analyze suite passes, including negative
resealed-protocol, source-cross-link, device, and runtime-lane gates. This
clears the local implementation and package-schema gates, not the Hopper
terminal-chain or 100-update cost gates.

The source-faithful JAX 0.4.31 stack detects the RTX 5090 but fails its first
JIT because its bundled `ptxas` predates Blackwell. Do not weaken the
Hopper/A100 lane to hide this. A separate JAX/JAXlib 0.6.2 lane with CUDA
12.9-era wheels reports compute capability 12.0. Its content-addressed
compatibility overlay mechanically replaces all 35 removed `jax.tree_map`
calls, and its exact 4-by-8/two-cycle/one-update CPU protocol agrees with the
JAX 0.4.31 reference to a maximum aggregate absolute error of
`5.960464477539063e-08`, with exact counters and checkpoint round trips.

Exactly one preregistered RTX 5090 PPO update was then attempted. Execution,
grouping, counters, and checkpoint structure passed, but the absolute-sum
aggregate for `params/fc_pi_1/bias` differed by
`2.0395550519458627e-04`, exceeding the frozen GPU absolute tolerance
`5e-05`; only 21 of 91 final leaves were byte-exact. The 5090 training gate is
therefore closed. Do not relax the threshold or treat this as benchmark
evidence. The next allowed diagnosis is non-updating comparison of rollout,
loss, unclipped-gradient, and clipped-update components to localize the first
discrepancy. Keep the upstream NumPy `<1.26` versus JAX `>=1.26` lane split
explicit, and retain Hopper/A100 as the source-faithful performance platform.

That non-updating diagnosis is now complete under a protocol frozen before its
single GPU capture. Initialization (92/92 records), task selection, observation,
reward, done, action, and minibatch-permutation streams are exact between JAX
0.6.2 CPU and the RTX 5090. The earliest failure is the cycle-one recurrent LSTM
carry: 9 elements exceed the unchanged tolerance, with maximum absolute error
`1.2597441673278809e-4`. PPO scalar loss terms, unclipped gradients, and global
norm/clipping remain within gate; both clip factors are exactly one. Near-zero
`fc_pi_1` gradients nevertheless differ enough that correct first-step Adam
normalization amplifies their aggregate update difference by about 29x. The
captured Adam proposal matches the analytic formula from the post-clipping
gradient, so this is forward/GEMM numerical drift rather than a sampler,
clipping, or optimizer implementation bug. No optimizer was applied and no
parameter changed. Keep training closed; the only justified next local step is
a separately frozen forward-only convolution/LSTM gate-preactivation trace.
That trace now passes through convolution, ReLU, flattening, embedding, and
feature concatenation; the first failure is the time-zero LSTM input GEMM.
All four gate slices fail, with stage maximum `1.825392246246338e-4`, while
propagating the measured feature drift through the exact kernels predicts at
most `4.74088485e-8`. This is strong evidence for default-precision backend
GEMM arithmetic rather than a semantic mismatch. A separately frozen
highest-precision forward-only dot probe then reduced every recurrent-stage
difference below its frozen threshold; final-carry maximum error fell to
`5.9604645e-8`, and CPU default/highest tensors were byte-exact. This supports
a separate two-line Blackwell compatibility overlay using `Precision.HIGHEST`
only on the LSTM input/hidden dots. That overlay must first reproduce the full
JAX 0.4.31 CPU one-update receipt and pass one separately frozen 5090
one-update gate; training remains closed until both succeed.
JAX documents this flag as a device-dependent accuracy/speed control for
float32 dot products: `HIGHEST` uses float32 on GPU, does not change the
input/output dtype, and has no impact on CPU backends. Thus it is appropriately
reported as a backend-numerics compatibility control, not a Frontier method
change ([JAX Precision documentation](https://docs.jax.dev/en/latest/jax.lax.html#jax.lax.Precision)).

The first CPU-only attempt of that two-line patch completed exactly one PPO
update and recovered excellent numerical agreement (546/546 aggregates,
24/24 floating statistics, maximum aggregate error `5.96e-8`, two cycles,
one update, one Adam application, and 64 trials). However, a wrapper-only
receipt write failed after the frozen update budget had been consumed. The raw
base receipt and both checkpoints were recovered read-only, but the required
outer provenance rewrite was never published. Under the predeclared budget the
gate is therefore `INCOMPLETE/HOLD`, not a pass; no 5090 update was attempted.
Keep this compatibility lane separate and repair the harness before any newly
authorized CPU/GPU protocol.

Hopper UED attempt 9366785 is retained as a failed infrastructure receipt:
all submitted hashes matched, but compute node `gpu021` lacked `git` and the
job exited 127 before import. The deterministic replacement pins Conda Git
2.45.2, fixes log-path resolution, and binds both import/JIT and exact
one-update engineering capabilities. Replacement import job 9366815 completed
in 48 seconds and its fetched source/environment/formula/JIT closure passed.
The exact dependent one-update job 9366863 then failed after 15 seconds, before
the driver or PPO update, because the compute node lacks `/usr/bin/time`.
No partial result tree was inspected. Remove that host dependency, rebuild the
immutable bundle, and repeat its exact-bundle import gate before retrying.
That replacement is bundle `6c2ca94ca8109be2775c`: import/JIT job 9366896
passed its full closure audit, followed by one-update job 9366897 (`0:0`, 1:42).
The latter verified one PPO update, `n_grad_updates=1`, exactly five optimizer
applications, 64 trials, four filled levels, zero incomplete/duplicate-new
groups, and checkpoint/Optax/PLR continuity. This clears the bounded
overlay-training rung only; the full terminal train/evaluate/assemble path and
100-update cost rung remain held. The in-process one-update receipt covers
16,384 transitions in 65.81 seconds (249 transitions/s, including first JIT)
with 1.74 GiB peak host RSS on a `1g.10gb` slice. Do not extrapolate that
startup-dominated rate to 30k updates: at face value it would imply roughly
23 days per run, which is precisely why a post-JIT 100-update cost pilot must
measure steady-state throughput before choosing MIG size, concurrency, or the
development budget. MAZE
full-arm cost job 9366552 completed successfully in 22:22, and its strictly
endpoint-blind cost/schema audit passed without opening or fetching any metric
endpoint. Peak GPU memory was 39,672 MiB, so retain the tested `3g.40gb` slice.
At the conservative paired standard deviation `.0135` and target effect `.005`,
30 blocks provide only about 38% power at the first Holm threshold `.025`, 60
provide about 71%, and 72 provide about 80%; prefer 72 balanced blocks if the
quota permits, or label a smaller freeze explicitly as a cost/precision
compromise. These are engineering and planning receipts, not benchmark
outcomes.

Keep BARN distinct. Its current groups choose one course and run all `N`
episodes there while the teacher pools a stratum posterior. That computes
`u_N(E[p_course])`, not the realized-course target `E[u_N(p_course)]`. Before a
BARN paper run, the clean repair would be to promote course identity to the
task unit or maintain hierarchical per-course posteriors and average
course-level utilities. A separate outcome-blind process froze the original
stratum-level design. Campaign `barn-icra2027-20260814-002` was canceled before
endpoint access after an unsupported publication operation was discovered;
replacement campaign `barn-icra2027-20260814-003` runs the same 20 sealed
Hopper tasks under the dated outcome-blind amendment. Do not modify it or
inspect partial endpoints.
Interpret that campaign, whatever its outcome, as a preregistered empirical
stratum-priority heuristic. It validates exact coefficient activity only under
an explicit within-stratum homogeneity assumption, which should be diagnosed
from sealed per-course outcomes after the full merge rather than asserted.
The current policy-only BARN learner also lacks the critic/GAE trajectory state
needed for canonical PLR, so any pass-rate replay baseline must be labelled
PLR-inspired rather than source-faithful PLR.

For a future exact robotics protocol, retain the useful ten-stratum interface
but make its score a two-stage posterior quantity. If requesting stratum `s`
still samples one of its courses uniformly and then runs all `N` episodes on
that course, maintain a separate Beta posterior `(a_c,b_c)` per course and use

```text
q_s = mean_{c in s} [1 - (b_c)_N/(a_c+b_c)_N - a_c/(a_c+b_c)].
```

This equals the posterior-predictive activity of the group that the adapter
actually purchases, including the shared latent course. It also handles unseen
courses transparently through the frozen prior. Compare this hierarchical
course-aware arm with the existing pooled-stratum arm and uniform under the
same transition budget; do not tune prior strength on held-out courses. Log the
selected stratum, selected course, pre-group course score, stratum mean score,
`K`, and realized `m_N(K)` so calibration can be checked at both levels. A
course-weighted second stage would define another method and must replace the
uniform mean above with its exact sampling-weighted expectation.

### P2. Complete E2c unchanged when its frozen GPU gate permits

Do not change the preregistered E2c protocol. The next stage is B1 seed 3, then
B2 seed 3, reservoir generation/preflight, three E2c runs, delivery validation,
and only then the sealed nine-arm endpoint.

Paper use is conditional:

- If delivery is valid and the seed-level raw-outcome pattern is coherent,
  include it as evidence about recycling direction versus replay.
- If delivery fails, call it treatment-delivery inconclusive and remove any
  suggestion that the direction term was isolated.
- If results are mixed, show all three paired seed contrasts and keep recycling
  secondary.
- Honor the Aug. 28 training hard stop already frozen in the project plan.

Do not run another gate sweep. GATE-DR has answered that question negatively.

### P3. After P1/P2, run one decisive LLM curriculum benchmark

This is the highest-leverage addition for a stronger paper after the existing
commitments. Current nearby methods report experiments on real reasoning models
and data: Learning-Zone Energy uses Qwen-family 1.5B--8B models, MoPPS covers
math/planning/geometry, and Actor-Curator learns policy-improvement-based prompt
selection. A 1.26M maze transformer alone will not fully answer the scale and
baseline concern.

Use one domain, not a new suite of disconnected pilots:

- a 1.5B-class reasoning model;
- a fixed, heterogeneous, train/evaluation-disjoint math pool with exact rewards;
- deployed `N` fixed across arms;
- identical initialization, prompt pool, update code, decoding, and paired seeds;
- primary budget in generated response tokens; report matched wall-clock and
  optimizer-update views as secondary;
- raw binary task outcomes at every evaluation.

Minimum arm set:

1. uniform;
2. `p(1-p)` with the same posterior, floor, and concentration;
3. exact `u_N` with the same machinery;
4. one source-faithful strong current selector, preferably LZE or MoPPS;
5. a mismatched score such as `u_2` or `u_64` at fixed deployed `N` if compute
   permits, to test peak-location specificity rather than generic hardness.

Pre-register a seed-level paired primary, SESOI, multiplicity family, and a
power/sensitivity analysis. Five seeds is a floor; use more if the pilot variance
shows that five cannot resolve the SESOI. Task bootstraps may quantify evaluation
noise but must not replace training seed as the independent unit.

Primary outcomes should be target-uniform AUC for mean@k and standard pass@k at
the same generation budget. Secondary outcomes should include out-of-distribution
held-out accuracy/coverage, selected-task entropy, realized coefficient activity,
dead groups, response tokens, and wall time.

### P4. Only after the core score is validated, improve the algorithm

Three extensions are principled and directly motivated by current limitations:

1. **Dependence-robust activity.** For arbitrary binary group dependence,
   half-mass is exactly
   `Pr(K>=1)-E[K]/N`. Estimate this discounted group-count quantity directly
   rather than reconstructing it from an i.i.d. pass-rate assumption. Compare it
   against the current Beta-Bernoulli score under controlled rollout correlation.
2. **Cost-normalized activity.** Rank by expected activity per predicted response
   token or simulator step. The present method prices a group but not its variable
   generation cost, which is the actual RLVR bottleneck.
3. **Recency-aware activity.** The current cumulative Beta posterior treats a
   task's success probability as stationary even though the student changes after
   every update. Old evidence can therefore make a once-hard or once-easy level
   remain overconfident long after it crosses the learning frontier. Compare the
   cumulative posterior with one prior-reverting discounted posterior,
   `a_t-alpha0 = gamma^(Delta t)(a_{t-1}-alpha0)+successes_t` and likewise for
   `b_t-beta0`, where `Delta t` is receipted outer-cycle age. The same analytic
   `E[u_N(p)]` remains valid for positive fractional Beta parameters. Freeze one
   half-life from an outcome-blind replay-cadence calculation, log effective
   posterior sample size, and give the plug-in posterior control the identical
   decay; do not tune `gamma` only for analytic Frontier or select it on the
   held-out test panel.

Do not add a learned gradient-norm or transfer model to the headline method yet.
That would blur the clean contribution and move directly into Actor-Curator/LZE
territory. First establish whether exact activity and cost normalization explain
enough of the gain.

## Experimental design upgrades that should become defaults

1. **Canonical budget:** generated tokens or simulator steps, with wall time and
   optimizer updates reported alongside. “Fixed steps” is insufficient when arm
   sequence lengths or probe costs differ.
2. **Fair selector sharpness:** report sampling entropy/effective support. Add an
   entropy-matched sensitivity so a win is not merely `u_N` being more or less
   concentrated under the same gamma.
3. **Held-out target distribution:** train selection can use a fixed pool, but
   the primary evaluation should include disjoint tasks/difficulties rather than
   only nested predicates on the training family.
4. **Treatment-delivery gates:** verify posterior calibration, selector
   separation, task exposure, and the intended `N` exponent before outcomes.
5. **Mechanism ladder:** predicted activity -> realized coefficient mass ->
   gradient/update norm -> held-out improvement. Failure at an earlier rung
   constrains interpretation of later performance.
6. **Independent unit:** paired training seed/block for inference; task and
   rollout resampling only for within-model evaluation precision.
7. **Raw outcomes:** retain per-task binary vectors, exact decoding seeds,
   checkpoint hashes, and evaluator/verifier hashes for every paper endpoint.
8. **Hyperparameter policy:** either tune every selector with the same
   outcome-blind development budget or use a shared setting plus a frozen
   sensitivity grid. Do not tune only the proposed method.
9. **Outcome branches:** keep the current practice of writing both branches
   before execution; add an explicit “claim removed” branch where appropriate.

## Manuscript redesign

### Main text

1. **Complete:** the title is **“Rollout-Aware Coefficient Activity for Task
   Selection in Verifiable-Reward RL.”** It names the supported method and
   avoids promising a general theory of recycling.
2. **Complete:** the abstract is roughly 180--220 words with at most two
   headline empirical numbers. It leads with the exact identity, the direct
   score test, and the neural-scale measurement; detailed caveats moved out.
3. **Open:** define the method once in a boxed algorithm with a convention
   table:
   estimator, all-fail handling, stabilizer, deployed group size, exact utility,
   posterior update, floor, gamma, and budget currency.
4. **Complete:** a compact contribution-to-evidence table separates the proved
   identity, scoped direct teacher, and engineering-only PLR overlay.
5. **Conditional:** make MAZE-SCORE the first empirical section only if its
   frozen evidence supports the score. Acrobot remains the current leading
   direct test; Digits is the counterexample.
6. **Complete:** the maze estimator factorial is supporting evidence for
   estimator-conditioned evaluation, not evidence for `u_N`.
7. **Complete:** paid-probe ProCuRL is appendix material and supports probe
   accounting, not method superiority.
8. **Complete for the current paper:** Countdown is absent from the contribution
   list; only the metric-provenance lesson is retained outside the headline.
9. **Complete:** procedural provenance is concentrated in the
   appendix/artifact while main text emphasizes scientific scope.
10. **Complete:** the acquisition-surrogate non-claim appears early and
    explicitly excludes expected learning progress, gradient norm, and optimal
    curriculum value.

### Related work and baseline position

The literature section correctly recognizes that the neighborhood is crowded,
but the experiment section does not yet meet the strongest neighbors. At minimum,
position and, in P3, compare against:

- [Group-standard-deviation identity](https://arxiv.org/abs/2607.00152): closest
  estimator-magnitude analysis on the normalized branch.
- [Actor-Curator](https://arxiv.org/abs/2602.20532): learned selection based on
  observed policy improvement.
- [MoPPS](https://arxiv.org/abs/2507.04632): closest Bayesian/Thompson prompt
  selection machinery.
- [Learning-Zone Energy](https://arxiv.org/abs/2605.17003): strong current online
  selector with 1.5B--8B reasoning experiments.
- [F-GRPO](https://arxiv.org/abs/2602.06717): coverage-preserving estimator-side
  alternative, important for the pass@k story.

The novelty claim should remain narrow: the exact unnormalized practical-MaxRL
utility and its rollout-budget-dependent peak, not posterior sampling, generic
intermediate difficulty, or universal estimator matching.

## Artifact and project engineering

1. **Complete for current artifact identity:** the local 53-row registry and
   release-only 562-row registry are explicitly distinct and are not silently
   substituted. A future release still needs a non-conflicting canonical path.
2. **Complete:** `reproduce.sh --build` targets
   `paper/main_iclr2027.tex` plus the extended wrapper, and the canonical build
   passed independent audit.
3. **Open:** make the reproduction command execute every central analyzer used
   by the compact paper: `N` sweep, Digits, Acrobot tournament, paid-probe
   analysis, maze block analysis, GATE-DR, E2c/MAZE-SCORE when available, and
   an automated claim-value comparison.
4. **Complete for deterministic local builds:** Python/package/font records and
   the Tectonic executable, mapping, index, format, and 483-member tree are
   content-bound; output-free regeneration and independent rebuilds match exact
   bytes. **Open for release:** publish a checksum-bound external bootstrap or
   container.
5. **Partly open:** have the build structurally fail if the conclusion exceeds
   page 9, citations are undefined, the ICLR-2027 style is not loaded, or the
   web PDF differs from the built current PDF. Citation and web-PDF checks pass
   now; add explicit page/style structural assertions rather than relying only
   on the frozen PDF hash.
6. **Open:** generate the claim trace from structured assertions rather than
   maintaining a manual Markdown table.
7. **Open:** vendor compact sufficient statistics and per-seed endpoints for
   every main claim. For large raw files, provide a stable download URI plus
   content and per-run hashes; a hash with no accessible object is provenance,
   not reproduction.
8. **Open:** mark stale synthesis documents (`EVIDENCE.md`, older
   readiness/review files) as historical or add a single generated
   `CURRENT_STATUS.md` to prevent superseded claims from resurfacing.
9. **Partly complete:** `docs/paper-iclr.pdf` is now published only from the
   exact verified compact build; record the corresponding source commit in the
   site when the artifact is intentionally released.
10. **Complete for current paper:** ICRA assets, smoke metrics, and
    preregistrations remain outside the ICLR run registry and paper figures.

## Go/no-go rules for the submission story

| Outcome | Recommended paper decision |
|---|---|
| Corrected MAZE-SCORE supports `u_N` over `u_2` and uniform | Submit the focused coefficient-activity paper; use this as the empirical anchor. E2c is optional strengthening. |
| MAZE-SCORE supports `u_N` over `u_2` but not uniform | Claim rollout-aware score-shape value, not a general curriculum win; keep uniform comparison prominent. |
| MAZE-SCORE does not support the primary | Remove neural effectiveness language for the sampler. Submit only if the diagnostic/boundary contribution is judged sufficient, or first complete P3. |
| E2c is delivery-valid and coherent | Keep a compact recycling/coverage result as a secondary consequence. |
| E2c fails delivery or remains blocked at the hard stop | Remove recycling from the title/contribution ladder; retain the historical proxy only as motivation for raw-outcome reporting. |
| Both MAZE-SCORE and E2c are negative/inconclusive | Do not force a broad ICLR story. Recast as a rigorous estimator diagnostic/boundary paper or complete the P3 LLM benchmark before a main-conference claim. |

## Schedule to the existing deadlines

- **Aug. 13--15:** verify the exact `u_N` dispatch, repair the MAZE-SCORE
  execution/analysis contract, and reconcile smoke/prereg hashes; do not freeze
  until the full launch gate passes.
- **Aug. 15--22:** after the cost receipt and clean outcome-blind freeze, run
  and analyze corrected MAZE-SCORE without endpoint peeking.
- **Through Aug. 28:** let the unchanged E2c driver run only when its frozen GPU
  safety gate authorizes; apply the hard stop exactly.
- **Aug. 29--Sep. 5:** select the outcome-appropriate paper branch, simplify the
  contribution ladder, reconcile artifacts, and rebuild the current PDF.
- **Sep. 6--12:** independent adversarial review focused on novelty, method-code
  identity, baseline fairness, and statistical units.
- **Sep. 16:** freeze title and abstract; leave buffer before the repository's
  recorded abstract/full-paper deadlines.

## Definition of “stronger paper”

The project is ready for a stronger submission when all of the following hold:

- one exact, canonical implementation maps configured `N` to the paper's `u_N`;
- a neural experiment directly tests `u_N` against `u_2` and uniform on fresh
  paired blocks;
- at least one competitive modern selector is tested under a fair shared budget,
  or the absence is explicitly accepted as the remaining ceiling;
- the primary method result uses held-out tasks, raw outcomes, and seed-level
  inference;
- proximal activity/delivery diagnostics accompany downstream performance;
- the abstract and contributions contain only claims carried by those results;
- a clean checkout can reproduce the current figures, statistics, page bound,
  and public PDF with one pinned command.

## Things not to do

- Do not reinterpret GATE-DR again or search more thresholds after its frozen
  negative verdict.
- Do not modify E2c, inspect partial sealed endpoints, or relax its GPU ceiling.
- Do not freeze MAZE-SCORE without a test and source receipt proving that its
  selected `frontier_un` path remains exact `u_N`.
- Do not claim ProCuRL inferiority from the 93%-probe protocol.
- Do not call the historical Countdown proxy standard pass@16.
- Do not count sampler observations, tasks, rollouts, or checkpoints as
  independent training replicates.
- Do not add the ICRA navigation smoke result to the ICLR evidence ladder.
- Do not add more loosely connected domains until MAZE-SCORE and E2c resolve.
