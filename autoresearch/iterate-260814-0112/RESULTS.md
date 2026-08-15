# Source and smoke ledger

## Reference target

The primary procedural-navigation target is the official `minimax` AMaze
benchmark. Published 30k-update mean OOD solved rates are approximately 0.55
for DR, 0.82 for robust PLR, and 0.83 for ACCEL. A later "beat" claim requires
the complete held-out suite, the same interaction/update accounting, paired
multi-seed inference, and wall-clock/resource reporting. These CPU smokes are
engineering evidence only.

## Pinned sources

| Artifact | Revision | License | Role |
|---|---|---|---|
| `facebookresearch/minimax` | `d053054c5290a04c1c4cd8b55704d999cad73e30` | Apache-2.0 | Executable AMaze benchmark |
| `facebookresearch/level-replay` | `ccecf452ee3342217ece964aaf10c2831625f9b3` | CC-BY-NC-4.0 | Algorithm/reference audit only |

The source-faithful clone is
`/data/robotixx/ued_bench/src/minimax-d053054`. The CPU environment is
`/data/robotixx/ued_bench/envs/minimax-jax0431-cpu`. A path-normalized,
sorted environment freeze has SHA-256
`a5a6ff4d57e44c282b4a73a4e54b1fa053f8235702269e71a34218ecec47b5b3`.

## Manuscript and artifact audit

An outcome-blind adversarial review classifies the current compact manuscript
as `HOLD` for competitive-method framing but `GO` for a focused theory/evidence
rewrite. The core coefficient identity, `T=N-1` deployment correction,
posterior expectation, and candid boundary results are defensible. The missing
empirical anchor is a direct neural score test: the existing maze factorial is
an estimator comparison, and the AMaze PLR implementation is materially
different from the paper's direct Thompson-sampled teacher.

The immediate artifact P0 is repaired. `bash reproduce.sh --build` now targets
the canonical `paper/main_iclr2027.tex` wrapper and validates its body plus the
compact registry. The manuscript now reports the actual checked-in 53 rows
(35 maze, 11 Countdown, 7 GSM8K) and explicitly distinguishes the untouched
release-branch 562-row object. The claim trace is contiguous from 1 through 62
with 35 exact and 27 rounded matches and no untraced row. Non-build
reproduction passed in a disposable copy. The subsequent pinned build uses an
exact Python/Matplotlib/font environment for isolated byte-comparison figure
regeneration and a completely hashed, network-free Tectonic 0.16.9 cache.
Two independent builds are byte-identical: the compact and website PDFs hash
to `36a6c1fb...` (15 pages), and the extended PDF hashes to `25023b85...` (25
pages). The checked-in receipt binds all embedded figures, toolchain members,
logs, warnings, and publication/rollback semantics without local path metadata.
An independent output-free rebuild and publication fault-injection audit found
no P0/P1/P2 issue; the exact audit record is `PAPER_BUILD_AUDIT.md`.

The subsequent focused rewrite aligns the narrative with that verdict. A
three-row contribution/evidence map distinguishes the theorem, the tested
direct teacher, and the engineering-only PLR overlay. The direct Acrobot score
test now appears before the explicitly secondary estimator-only maze result;
paid-probe, recycling, Countdown, and gate material is confined to the
appendix. The text makes no AMaze, robotics, PLR, ACCEL, or `minimax`
performance claim. All 62 trace rows and static LaTeX references remain valid.

## Smoke receipts

### Domain randomization

- Path: `/data/robotixx/ued_bench/runs/dr-cpu-smoke-d053054-seed1/smoke`
- Configuration: seed 1, 1 update, 2 environments x 8 steps, reduced 5x5
  maze/model strictly for compilation and I/O validation.
- Wall/peak RSS: 14.74 s / 746368 KiB.
- `checkpoint.pkl`: `f57d707d4514dbed54d064fbc658f670697010419f931509e1a7e0a4db8fe202`
- `logs.csv`: `2d2dcfa18097ac38d521a088861ffd1d0ba126b70ebd6b78ad621ea317fe177c`
- `meta.json`: `2edb6d66d3da5b5c7a079007e4734880c8a3f207c7e5b7b494dfb59fdb916c3a`

### Robust PLR

- Path: `/data/robotixx/ued_bench/runs/plr-cpu-smoke-d053054-seed1/smoke`
- Configuration: seed 1, buffer 4, minimum fill 0.5, replay probability
  1.0, rank temperature 0.3, staleness 0.3, forced uniqueness. Two new-level
  evaluations warm the buffer; three robust replay PPO updates then execute.
- Total: 64 environment steps; 21.84 s wall; 966232 KiB peak RSS.
- `checkpoint.pkl`: `8ff8fd8a61751fb38161f379b97aa89ed3297725c9238d0ad5852bf112fdb156`
- `logs.csv`: `a25e9c7e975de95311dba2cadb961121c6fb14002c46899f8cf6a96443eed7c5`
- `meta.json`: `7dd84cbbc4456d76fa1cdf28105fa77974c2b28c8495b344138c0f834c8e8f92`

## Intervention contract

The first FrontierRL arm changes only the level score. It estimates each
buffered level's sparse-goal rollout-success probability with an online Beta
posterior and ranks levels by

`u_N(p) = 1 - (1 - p)^N - p`.

Two labels are required. `Posterior-Frontier-PLR` keeps upstream `n_eval=1`,
accumulates ordinary visits, and adds no probe rollouts, but its `N>1` utility
is counterfactual to the current PPO batch. Exact grouped coefficient activity
uses `n_parallel=4,n_eval=8,N=8`, preserving 32 simultaneous rollouts and total
student transitions while evaluating four tasks eight times each. Both retain
PPO, robust replay gating, replay probability, rank transform, staleness,
buffer capacity, architecture, level generator, and evaluator. The prior,
`N`, and actual `n_eval` are explicit factors. Grouped MaxMC and the upstream
32x1 MaxMC arm are separate controls.

## Final local Frontier overlay receipt

The current overlay contract is
`5868d346ba00e43225424f053ded3dea056b8f76ce4bfdaa1a107a4955cb9000`.
The frozen workspace bundle digest is
`fd9aa789f5a015cc2d7478264bc4625c02ba1742755fdaa298a8894e8ec31562`;
the independently applied clone is
`/data/robotixx/ued_bench/src/minimax-frontier-v3-final-5868d346-d053054`
with applied manifest
`d929efa2f059a93125e217ec4713ae81670c769d979c67abd2b10efc64268af3`.

The final suite passes 20/20 checks covering the analytic score, grouped
stream completion, duplicate evidence, buffer insertion and eviction,
checkpoint signatures, parsed 4-by-8 LSTM geometry, posterior insertion, and
a real PPO update. Exact grouped Frontier and the 32-by-1 bridge both complete
three full-horizon PPO updates; exact Frontier resumes to update four and its
posterior trial total advances to 160. A late adversarial regression caught a
full-buffer identity bug in which a new-level eviction could overwrite a slot
before a later existing-level observation used its stale index. The corrected
ordered update leaves the concrete mixed batch as `{99: 2/4, 20: 2/4}` and
never creates the contaminated `2/8` posterior. All earlier overlay contracts
are revoked.

A separate two-test RNG-layout contract exercises the pinned `BatchEnv` and
TensorFlow Probability categorical sampler. Within each four-by-eight batch,
all eight copies share the intended level reset key/state; the 32 step keys
are pairwise distinct, and a fixed-key equal-logit policy draw is not broadcast
as one scalar action. Both tests pass. This is the mechanical prerequisite for
conditionally independent evidence, not a test that realized successes are
independent.

The primary score-isolation pair is now exact Frontier versus MaxMC with the
same 4-by-8 grouping and 500-level buffer. With minimum fill 0.5, both begin
training after 63 fill cycles: 2,016 evaluation-stream slots and 516,096
transitions. The official 32-by-1/4,000 MaxMC arm remains a separate
source-faithful reference rather than an unfairly warm-started control.

The audit also identified a method-level nonstationarity risk before any AMaze
development endpoint was opened. The cumulative Beta posterior treats each
level's success rate as stationary while PPO continually changes the student;
high historical trial counts can therefore make the estimated frontier lag.
The improvement plan now includes one outcome-blind, prior-reverting discounted
Beta ablation with a replay-cadence-derived half-life and the same analytic
expected-activity formula for fractional counts. This does not alter or unfreeze
the current cumulative-posterior score-isolation protocol.

A cost-matched mechanistic follow-up was also derived outcome-blindly. For
`N={2,4,8}`, choose `n_eval=N`, `n_parallel=32/N`, and
`buffer_size=4000/N`, and pair Frontier with MaxMC inside each layout. All six
cells retain 32 rollout streams and approximately 63 all-new warm-up cycles
(516,096 student transitions). This equality requires distinct successful new
insertions; any duplicate-new group delays fill and must be exposed in the
receipt rather than silently counted as a nominal warm-up cycle. A within-N
score contrast followed by a
contrast-of-contrasts can therefore test rollout-budget dependence without the
known replay-onset confound. This is planned, not executed, and must be frozen
before any N=8 development endpoint is opened. Five-seed interactions are
descriptive only (minimum nonzero exact two-sided sign-flip `p=.0625`); any
paper claim requires untouched confirmatory seeds or a pre-frozen interaction
family with multiplicity control.

The upstream inverse-rank transform exposed a more immediate pre-endpoint
blocker. It breaks exact score ties by stable sort order and then raises inverse
rank to `1/temp`; at the frozen `temp=.3`, rank 1 holds about 87.16% of the
score-only mass and the top five hold 99.31%. A newly evaluated N=8 slot has
only nine possible posterior scores, so buffer position can determine replay
inside large tie blocks. Development performance is now held pending a
tie-aware, block-mass-preserving rank transform applied to both Frontier and
its group-matched MaxMC control, with the untouched upstream MaxMC retained as
the source-faithful reference. The current terminal and cost smokes remain
valid engineering gates because they are not method-performance evidence.
A standalone NumPy prototype passes block-mass, permutation, distinct-score,
and all-equal invariants: for three tied top scores in a five-slot example,
upstream assigns `[.87764,.08707,.02254]`, while the proposed transform assigns
`.32909` to each and preserves their combined mass exactly. No overlay code or
performance endpoint was changed by this prototype.

The exact vectorized construction was also JIT-checked under the pinned CPU
runtime (`JAX/JAXlib 0.4.31`). Stable sorting, fixed-order segmented scans, and
canonical tie-mode normalization preserve every filled tie block's upstream
mass within float32 tolerance, give equal members equal mass, leave
distinct-score distributions mathematically unchanged, and assign zero mass
to unfilled slots. Exact normalized bit parity for every permuted
distinct-score buffer is impossible while also removing slot-order-dependent
float32 normalization; the v4 contract therefore keeps the default path
bit-identical, makes the opt-in normalized probabilities permutation
equivariant, and bounds the distinct-score difference by a frozen float32
tolerance. This establishes implementation feasibility without modifying the
frozen v3 overlay.

The event identity also yields a predeclared mechanism target for the next
overlay. For observed group success count `K`, realized coefficient half-mass
is `1{0<K<N}(N-K)/N`, whose conditional expectation is exactly `u_N(p)`.
The v4 telemetry should therefore bind each pre-update posterior prediction to
this subsequently realized value and report frozen-bin calibration/error for
both Frontier and the group-matched MaxMC priority. This is a direct test of
the scoring claim and does not depend on opening the held OOD performance
endpoint.

A follow-up code audit established that `force_unique=true` does not resample
duplicate replay tasks. `_sample_replay_levels` draws sequentially with
replacement, and `dedupe_levels` only masks or consolidates PLR-buffer updates;
the PPO batch still contains every sampled group. Frontier intentionally
accumulates repeated existing-level Bernoulli groups. For a 500-slot pure
inverse-rank distribution at temperature `.3`, four draws contain only
`1.4676` unique levels in expectation, and all four are rank 1 with probability
`.5770` (before the 0.3 staleness mixture). In the five-slot three-way-top-tie
prototype, tie-aware weighting raises expected unique levels across four draws
from `1.4427` to `2.4426` and lowers the probability that all four draws are
identical from `.5934` to `.0352`. These are analytic delivery diagnostics,
not benchmark outcomes. The v4 receipt must log realized unique replay levels,
duplicate groups, and effective support; a separate without-replacement
ablation must not be folded into the score-only primary.

At the exact 252-slot first-replay point, a further prior-predictive stress
test assigns 28 fresh Beta(1,1), N=8 levels to each possible success count
`K=0..8` (the Beta-binomial count is uniform). The `K=2` top-score tie retains
`.999850` score mass under either rule. Stable ranks concentrate `.871570` on
one slot (ESS `1.3026`, expected unique-of-four `1.4676`); block-mass-preserving
ties spread it equally across 28 slots (ESS `28.0084`, expected
unique-of-four `3.7908`). This is a constructed pre-endpoint diagnostic, not
an empirical AMaze buffer or performance observation.

The resulting v4 engineering overlay is finalized at contract
`3d5f3827a82a4f713314091289196a1c2909dd5d7c4c96dd532052c5706e832b`,
applicator
`c2e5eb3dac02b86723ece485cd348832f1636198c781bae82c1d99df0167590b`,
tie module
`1b9db20d05edd3212346e84d14606af91ae443c0665945a7b679ade161560244`,
lineage
`784e2fd1f545d49c8d10c3f3aeda37aae51fa00127e2c14578702e275bfb6971`,
and applied manifest
`9b411f61ebc56bb93fc22cad6b19299c38eab2b696fa17f7783c7729e1db02ae`.
It contains no scatter reduction in source or JAXPR. Independent pinned-JAX
audit passed 34 core/legacy tests, the bounded one-update E2E, six
terminal/two-arm tests, exact actual-distribution permutation checks, default
distribution and sampling parity, cross-version rejection, lineage
reconstruction, and checkpoint/snapshot closure. Across 5,000 distinct B=500
permutations, the maximum normalized source-stable versus tie-mode rounding
difference was `3.57628e-7`, within the frozen `5e-7` limit. No performance or
OOD endpoint was read. The separate v2 protocol remains DRAFT and explicitly
sets `production_driver_authorized=false`; this receipt is not paper evidence.

The sibling-only v4 Hopper tooling has a separate deterministic local bundle:
ID `d602ce7854f8f3e99352`, root manifest
`d602ce7854f8f3e99352025b97eed2fde32733c0dd23297d5c28b1051e7aeaf0`,
state `7a56b89e...`, and overlay manifest `cf600946...`. Twin builds are
byte-identical. Local R0 import, R2 Frontier one-update, R3 two-arm terminal
Phase-A/B with 60 sealed evaluation records, and a fresh-clone 22/22 pinned-JAX
suite pass. Independent audit grants only local engineering GO: no remote
stage/submit path is authorized until exact Hopper export, MIG, `sacct`, fixed
Conda, and manifest-bound Phase-B interpreter checks close. Cost100,
production, analyzer, endpoint, and paper-evidence paths remain disabled.

A bounded follow-up hardens the proposed remote contract without authorizing
it. Its deterministic twin bundle is `da74eb3e0debc7781d6d`, root manifest
`da74eb3e0debc7781d6d785f9406acec953a02cfcc3674afeb70c0f438619cc8`,
with state `56ee7c1b...` and overlay closure `c923440e...`. Seven focused
adversarial tests, Slurm-spool/tamper simulation, syntax checks, and exact
source-to-twin closure pass. The independent verdict is narrow local-contract
GO but full/remote/paper HOLD. Four primary remote blockers remain: the widened
overlay contradicts the preserved d602 R1/R2 closure, R2 disagrees on
`job-<id>` identity, JAX/GPU probing starts under system Python, and missing
MIG `gpumem` accounting is incorrectly rejected instead of recorded as
unavailable. Additional post-terminal, trust-anchor, Phase-B, and real-Hopper
schema gates are listed in
`autoresearch/iterate-260814-v4-remote-hardening/RESULTS.md`. No remote action
was taken.

The rollout-count follow-up is now a hash-closed six-cell DRAFT rather than an
informal proposal. Frontier and MaxMC are paired within `N={2,4,8}` using
`n_eval=N`, `n_parallel={16,8,4}`, and `buffer_size={2000,1000,500}`. This
keeps 32 streams and 8,192 training transitions per outer cycle. The protocol
is `81a57668d3cfdf595f13710df6152a437b8c4640791fbeeed2ef8c9e9486f26f`;
the six-config closure is
`58e1ffd9c7e3d80992971b331c540d6c8976c9cd4082391fae92de0df4fd417f`.
Independent pinned-CPU validation passed 12/12 tests, parsed all six cells,
produced six unique grid-generated XPIDs, and rejected unsafe direct
`xpid=latest`, cross-`N` checkpoints, and strict Frontier `N/n_eval` drift.

Within each `N`, the exact update, upstream-gradient, optimizer-application,
outer-cycle, transition-grid, and terminal-checkpoint receipts must match
before computing a paired score contrast. Across `N`, the estimand is joint
effect modification by `N/n_eval/n_parallel/buffer_size`, not a pure
coefficient-estimator `N` effect. The common 63-cycle/516,096-transition warm
fill holds only if every fill cycle accepts the required distinct new groups;
duplicates, partial cells, or rejected groups force reporting the observed
fill trajectory instead. No performance, OOD, Hopper, or GPU endpoint was
accessed, and the DRAFT explicitly authorizes none.

An outcome-blind construction check against the pinned AMaze registry also
resolved the future twelve-maze confirmatory budget. In frozen panel order the
maximum horizons are `[450,450,450,450,450,450,450,484,250,450,882,882]`,
so ten episodes per maze cost at most 60,980 external-evaluation transitions
per terminal checkpoint. The later confirmatory evaluator must bind this full
vector rather than reuse the three-maze development constant of 450.

The confirmatory ten-seed count is now treated as a floor rather than inherited
from the source paper. As a planning approximation, detecting an absolute
solved-rate SESOI of `.02` with paired SD `.03` needs about 18 pairs for 80%
power at two-sided alpha `.05`; SD `.04` needs about 32, before small-sample or
multiplicity allowance. The final count remains unfrozen pending the
development variance and cost receipts, and must be fixed outcome-blindly
before any confirmatory endpoint.

The source benchmark target was made explicit from minimax Table 2: after 30k
PPO updates, its ten-run LSTM means are robust PLR `.82±.02` and ACCEL
`.83±.02` on the full OOD maze suite. A future “beat minimax” statement now
requires a direct, current-code, matched ACCEL rerun and a positive paired
interval on the frozen twelve-maze panel; a three-validation-maze score or an
unpaired point estimate above `.83` is insufficient.

## Paper-theory hardening

For a buffered task with `p ~ Beta(a,b)`, the deterministic posterior
Frontier priority is now stated in the manuscript as

`E[u_N(p)] = 1 - (b)_N/(a+b)_N - a/(a+b)`.

The corresponding plug-in optimism is exactly

`u_N(a/(a+b)) - E[u_N(p)] = (b)_N/(a+b)_N - (b/(a+b))^N >= 0`.

`curriculum_maxrl.test_mass_formulas` independently checks the expectation by
256-point Gauss--Legendre quadrature for six Beta/group-size settings and
checks both Jensen ordering and the printed gap. The complete coefficient
suite remains green. This is an analytic result, not a benchmark outcome.

The compact ICLR-2027 narrative was narrowed without changing any quantitative
claim: the abstract now centers the exact identity, Acrobot, the Digits boundary,
and estimator-conditioned maze result; Countdown is no longer a headline or
contribution. The canonical title is now *Rollout-Aware Coefficient Activity for
Task Selection in Verifiable-Reward RL*, and the claim trace records this
narrative-only revision. The ICLR-2027 style file is present, a simple source
count puts the abstract at 185 whitespace-delimited tokens, braces balance to
zero, and the 14 environment begins match 14 ends. The pinned cached Tectonic
path compiles both wrappers and reproduces the exact checked-in PDFs; an
unrelated-host bootstrap for the checksum-bound external tool assets remains
an artifact-release requirement.

## Matched-development analysis gate

`ued_benchmark/UED_MATCHED_DEV_PREREG.md` and the fail-closed analyzer freeze
the first exact 4x8/buffer-500 Frontier-versus-MaxMC development comparison:
training seeds 101--105, three validation mazes, ten episodes each, paired
evaluator seed `100000+s`, and a positive mean/no-environment-regression-worse-
than-`.05` keep rule. The endpoint is 30,000 student PPO updates. Both upstream
counter fields must equal 30,000, while the true optimizer-step count is
separately reconciled as 30,000 x 5 epochs x 1 minibatch = 150,000.

The analyzer rejects periodic checkpoints, preterminal state, resume, budget or
hash drift, missing terminal Slurm accounting, and incomplete ten-run matrices
before reading metrics. A hashed loop-owning driver now bypasses upstream
`ExperimentRunner.train`'s nonterminal-checkpoint ambiguity, while a separately
hashed evaluator fixes seed, maze order, backend, and device count and retains
all 30 per-episode outcomes. The driver emits the missing per-slot analytic and
plug-in scores, counts, ages, identities, and replay probabilities into a
separate manifest-closed sidecar. It also checks reconstructed scores and replay
mass against the pinned implementation. After correcting evaluation accounting,
ten bounded driver/evaluator tests pass; actual CPU evaluation for both
arms agrees with an independent upstream `EvalRunner.run` across all six fields
within `2e-6`. Each frozen validation maze resolves to 450 steps, so external
evaluation costs 13,500 transitions rather than the earlier 7,500 estimate.

The schema-2 assembler is now atomic and embeds both manifest-closed source
packages, their receipts, the 30 ordered raw records, run context, terminal
logs/accounting, and Frontier snapshot. Both runtime drivers verify the exact
campaign digest and its protocol/analyzer/assembler/driver/provenance bindings
before importing `minimax`; the evaluator independently cross-links the
trainer protocol, execution mode, and fresh source receipt. The complete
bounded train/evaluate/assemble/analyze suite passes 31 tests, including
resealed-drift, symlink, traversal, device, and partial-package negatives.
Launch remains held until the same two-phase terminal chain passes Slurm and a
100-update cost/package smoke measures steady-state performance.

## GPU engineering receipts

### RTX 5090 modernization probe

The pinned JAX 0.4.31 source-faithful environment detects the Blackwell GPU
but fails its first JIT because its bundled `ptxas` cannot compile the target.
A separate, non-source-faithful probe at
`/data/robotixx/ued_bench/envs/jax062-cuda129-probe` uses JAX/JAXlib 0.6.2,
CUDA 12.9-era wheels, Flax 0.10.7, Optax 0.2.5, and an isolated sorted freeze
hash of
`0a9ae6498457aafd71cde1c70f817bdad9c84517b25ad4b639e08e8538e61ec9`.
With preallocation disabled, it reports the RTX 5090 at compute capability
12.0, runs a 1024-by-1024 matrix JIT in 0.693 seconds, and evaluates the
Beta(3,2), `N=4` analytic activity as `0.328571379` versus exact
`0.328571429` (absolute float32 error `4.94e-8`).

A disposable clone at
`/data/robotixx/ued_bench/src/minimax-frontier-blackwell-jax062-5868d346-d053054`
retains the final Frontier overlay and applies a separate two-line patch,
SHA-256
`8d41711e7babf9b7d7e6f8242fc05d23e26a39bf3950c3f26a534f2f4e0d2528`,
that replaces `jax.tree_map` only in `src/minimax/envs/environment.py`. On the
5090, a genuine AMaze reset and one step then compile and execute: 5-by-5-by-3
observations, state time 1, reset compile/execute 1.051 seconds, and step
compile/execute 1.345 seconds. Three static contract tests and `pip check`
pass. The canonical overlay and source-faithful clone are unchanged.

This first probe cleared AMaze import/reset/step only, with zero training
updates and zero benchmark episodes. A subsequent content-addressed
compatibility overlay mechanically replaced all 35 remaining `jax.tree_map`
calls across ten files. Its exact 4-by-8 Frontier protocol performs two outer
cycles and one PPO update. On CPU, the modern JAX 0.6.2 lane starts with 91 of
91 leaves byte-exact against JAX 0.4.31 and finishes with maximum aggregate
absolute error `5.960464477539063e-08`; counters, 117-leaf checkpoint
structure, pickle round trip, and fresh-runner resume all pass. The overlay
contract is
`b7c865e007634c5a20e2b942ff98f24d6ac9ff624d5b17b62e5e9fa2124e5c00`.

Exactly one RTX 5090 PPO update was authorized after that CPU gate. It ran and
saved structurally valid checkpoints with two cycles, one PPO update, 64
posterior trials, four filled levels, and zero incomplete or duplicate groups.
A CPU-only readback nevertheless found one aggregate outside the frozen GPU
tolerance: `params/fc_pi_1/bias` absolute sum differed by
`2.0395550519458627e-04` versus `atol=5e-05` (`rtol=5e-04`); 21 of 91 final
leaves were byte-exact. Recovery receipt SHA-256 is
`cad634ba29f3455a2cce5af383414f3ff937564487d51e9bb59b36652fd4d446`.
The GPU gate failed closed: no retry, OOD evaluation, multi-seed run, longer
training, or tolerance change was made. The next safe investigation is a
non-updating rollout/loss/gradient/clipping comparison. Upstream also pins
NumPy below 1.26 while JAX 0.6.2 requires at least 1.26, so this remains a
separately labelled compatibility lane rather than source-faithful evidence.
The complete machine-readable outcome manifest has SHA-256
`7326be4118238e0c25eddac2fa3f985a1cecc5adfc3c0c2d0e9892a9f475ce47`.

A later component-parity protocol was frozen before one non-updating 5090
capture (protocol SHA-256
`0f8c083202a189ec234f32c0e1c15e7c09753892fb05af0d6262b9ff0bf9f1a5`).
It compares JAX 0.6.2 CPU and GPU with the same frozen initial checkpoint and
applies zero optimizer updates or parameter mutations. Initialization, task
RNGs/levels, observations, rewards, dones, actions, and the PPO minibatch
permutation are exact. The earliest failure is the cycle-one recurrent LSTM
carry: 9 elements fail the unchanged GPU gate, with maximum error
`1.2597441673278809e-4`. Scalar loss, unclipped-gradient, global-norm, and
clipping stages pass; CPU/GPU clip factors are both one. Correct first-step
Adam normalization of a near-zero policy-hidden gradient amplifies the
upstream difference by roughly 29x, but captured proposals match the analytic
post-clipping formula, ruling out a clipping or Adam implementation mismatch.
The primary comparison SHA-256 is
`09b4745799e689e62c0b68db900947fdd55a2ab72cb80747721b6067e48ae2d5`;
the closed manifest SHA-256 is
`4ff48dab1a3a6fe03384db7254229e07cb8bf6576d68fff4a74c5a95500483af`.
This diagnoses `forward_or_gemm_recurrent_carry` and leaves the training gate
closed.

A second protocol was frozen before one forward-only 5090 capture (protocol
SHA-256
`024239a6b659097198a6d902b1bb63698849d38e340ac033fa21537b0e5888ce`).
Convolution, ReLU, flattening, embedding, and concatenated features pass. The
first failure is the time-zero `OptimizedLSTMCell` input GEMM: all four gate
slices fail, with input-affine stage maximum
`1.825392246246338e-4`. The observed feature discrepancy is only
`5.9604645e-8`; multiplying that perturbation by the exact kernels predicts at
most `4.74088485e-8`, 3,617--4,437 times below the observed affine error. This
supports default-precision backend GEMM arithmetic rather than a semantic
mismatch. No gradient, optimizer proposal/application, mutation, OOD, or
performance endpoint ran. The comparison SHA-256 is
`1716d6ccbecaa57bf7babb4028e48bc4a8914efd6546afd847158e735d1e2927`
and the closed manifest is
`0137df6ee42bf6731370d1ec53fb06df3ffa2b4f35d55f68ea283d72e0c91d0c`.

A third, separately frozen forward-only precision probe changes only the LSTM
input and hidden dot precision. `Precision.HIGHEST` passes all required
cross-backend recurrent stages: input affine `8.9406967e-8`, hidden affine
`2.9802322e-8`, activation `1.1920929e-7`, cell `5.9604645e-8`, hidden
`2.9802322e-8`, and final carry `5.9604645e-8`. CPU default/highest records are
byte-exact. The protocol SHA-256 is
`0abdb46a7b56986756a31f3d4cc1793af20fc6ca53d2b397720386aab7f5b820`
and comparison SHA-256 is
`e1f6034d2ed66492dd2f0df45d93515eea1852d94ff747763e4f10abf8f86f6f`.
This justifies CPU-first testing of an isolated two-line compatibility patch;
it is not itself a training or performance result.

That isolated patch subsequently completed exactly one CPU PPO/Adam update.
Read-only recovery of its raw base receipt and checkpoints verifies 546/546
aggregate comparisons, 24/24 floating statistics, 91/91 exact initial leaves,
two cycles, one update, and 64 posterior trials; maximum aggregate error is
`5.960464477539063e-08`. The wrapper nevertheless failed after the update while
writing its required outer provenance receipt. Because the frozen CPU update
budget was exhausted, it was not rerun: numerical parity is green, the wrapper
gate is `INCOMPLETE`, and the RTX 5090 gate remains `HOLD/not attempted`.
This is a harness result, not a benchmark or performance endpoint.

The probe initially placed 287,548,507 bytes of new pip-cache entries on the
nearly full root filesystem. Those 103 timestamp-bounded cache files were
removed; they are recoverable by redownload. All subsequent cache and
temporary paths were redirected to `/data/robotixx/ued_bench`. A later
dependency resolver added at most roughly 80 MB to the pre-existing pip cache
before that redirection reached the parallel probe; those entries were left in
place because concurrent ownership was ambiguous.

### Hopper A100 attempt 9366785

The first bounded A100 import/formula/JIT attempt used immutable bundle
`2afc40909990229f8c86`, bundle manifest
`2afc40909990229f8c86756ee0dba77dff46681154273f3ab76776ccbb106d52`,
and a directly constructed environment at
`/scratch/lwang44/envs/ued-minimax-v2-9ab83896f41c5294-6eb6a5a4d12697fd`.
The environment lock, setup, freeze, and manifest hashes all verified before
submission. Independent review also verified the local/remote sbatch and all
export bindings.

Job 9366785 ran on `gpu021` from 02:48:49 to 02:49:04 EDT and terminated
`FAILED`, exit `127:0`, before importing the benchmark. Its engineering-only
log contains the single causal error `git: command not found`; Hopper compute
nodes do not expose the login node's `/usr/bin/git`. There is no qualified
`COMPLETE` result and no scientific endpoint. Preserve this attempt as failed
infrastructure evidence. The replacement bundle must pin Git inside the
environment, invoke that exact binary, rerun local content-addressing tests,
and pass a fresh import/JIT job before any one-update job is eligible.

The replacement is immutable bundle `e675359647be418bd800`, manifest
`e675359647be418bd800ba80085e3d26973436a554abeca3730059e9e8fe4a64`,
with overlay manifest
`c032e80c560f3e82533ffc825f8884306d976946d315c38f58901d0cf9885e01`.
Its new environment is
`/scratch/lwang44/envs/ued-minimax-v2-9ab83896f41c5294-dbd0494789fd70b8`;
setup SHA-256 is
`dbd0494789fd70b8a2d677e0341ec8feab7623ade3133677cdf176dc75dcac2e`,
freeze SHA-256 is
`10dfc24e1531c81bd6e788dbfec8d003cd66b0427594de4bd54bc7d5d772105f`,
and environment-manifest SHA-256 is
`a75ac3eeb3964ffde1ba71e194f050e50377864ca3bf83393152813877f552a9`.
It pins Conda Git `2.45.2=pl5340h9abc3c3_0`; the import and one-update scripts
invoke that exact binary. Replacement job 9366815 completed `0:0` in 48 seconds
on `gpu021`. Hardened retrieval to
`/data/robotixx/ued_bench/hopper/import-smoke-job-9366815` verified its complete
tree, source-faithful import, A100 MIG backend, both Frontier formulas, and one
JIT. The result-manifest SHA-256 is
`7416b652ed46963e903ea438a2c5204d6574db8356d62b5ee9388a1c5e46c307`.

The bundled one-update gate is limited to two outer cycles, 16,384 student
transitions, and exactly one PPO update. Local staged E2E tests pass twice. It
must consume the completed 9366815 manifest from this exact bundle and must
assert 64 trials, zero incomplete/duplicate-new groups, checkpoint reload
continuity, and no OOD evaluation before it is eligible to publish `COMPLETE`.

Exact dependent job 9366863 carried those bindings but failed `1:0` after 15
seconds, before the grouped driver or PPO update. The bounded terminal log
shows the sole cause: compute node `gpu021` lacks `/usr/bin/time`. No partial
result tree was inspected and no retry was submitted. The replacement will use
Python standard-library timing/resource accounting plus terminal Slurm `sacct`;
because the sbatch/driver changes, it requires a newly content-addressed bundle
and a fresh exact-bundle import/JIT gate.

That second replacement is immutable bundle `6c2ca94ca8109be2775c`, manifest
`6c2ca94ca8109be2775ce0f166e11f064466e4aaa3c2efb085587a0d3f13e93d`.
Fresh import/formula/JIT job 9366896 completed `0:0` in 45 seconds; hardened
retrieval to `/data/robotixx/ued_bench/hopper/import-smoke-job-9366896`
verified all 17 bundle payloads, five environment payloads, the source-faithful
JAX 0.4.31 A100-MIG backend, both score modes, and one JIT. Its result-manifest
SHA-256 is
`3a15f52ddb0aa0b44f190f9701183c51884b91a0f1d850f327a53c3208f2a14c`.

Dependent job 9366897 consumed that exact closure and completed `0:0` in 1:42.
The audited result at
`/data/robotixx/ued_bench/hopper/one-update-job-9366897` has manifest
`4eaa676052cbc9006da1d285b03eda354cab27f3b7d72064b5138724c83691c8`
and tree digest
`e5f32761a3ee2b0ed25a9f8637f066b88a445769f037af1d579aefe376fed3d3`.
Cycle one ended with 32 trials and no PPO update; cycle two ended with exactly
one PPO update, `n_grad_updates=1`, five optimizer applications, 64 trials,
four filled levels, and zero incomplete/duplicate-new groups. Independent CPU
unpickling reconfirmed the checkpoint, Optax count, PLR state, static signature,
and exact leaf continuity. No post-resume update, OOD evaluation, or paper
endpoint ran. This clears the bounded overlay-training rung only.

The Python resource receipt measured 65.807 seconds for 16,384 transitions
(248.97 transitions/s including first compilation) and 1,824,760 KiB maximum
host RSS. This is not a steady-state estimate: naively extending it to the
approximately 492 million transitions expected per 30k-update robust-PLR run
would imply about 23 days. A production-shape 100-update pilot must separate
JIT startup from steady-state cycle time before any campaign resource request
is frozen.

The full terminal train/evaluate/assemble path is now frozen separately as
engineering bundle `06ffeeeb6998e8ddb1ce`, manifest
`06ffeeeb6998e8ddb1ce516c8982ef8e78627f7cc876ea0b712dab466aa1e8ff`.
Two independent builds are byte-identical and checksum-clean. Independent
audit found no P0/P1/P2 after exercising exact Slurm/resource identities,
one-update/128-training-transition limits, actual non-synthetic
`3*10*450=13,500` external-evaluation transitions, post-terminal-only fetches,
authoritative `sacct`, isolated Phase B, atomic assembly, and permanent
`paper_evidence=false, analyzer_eligible=false`. The exact bundle and unchanged
environment are staged and verified remotely; fresh import/JIT job `9367063`
is pending for scheduler priority. No partial stdout or result has been read.

## Endpoint-blind MAZE cost gate

Hopper job 9366552 completed one full 250-update arm in 22:22 on `gpu013`.
Only its safe accounting/schema package was retrieved: the result JSONL,
telemetry, checkpoint, stdout, and all metric values remain unopened. Peak
host RSS was 1,368.47 MiB and peak GPU memory was 39,672 MiB on a `3g.40gb`
slice, so the smaller MIG profile is unsafe without a new engineering smoke.
The output schema contains one config row plus the exact evaluation schedule
at updates 0,25,...,250 and passed full-tree checksum validation.

At the conservative paired SD `.0135`, SESOI `.005`, and first Holm threshold
`.025`, approximate power is 38.2% for 30 blocks, 71.4% for 60, and 80.1% for
72. A strongest-paper freeze should therefore use 72 balanced blocks if quota
allows; 30 or 60 must be described as a cost/precision compromise. No MAZE
evidence job is authorized while its preregistration is draft and its current
engineering bundle is dirty.

Separately, frozen BARN campaign `barn-icra2027-20260814-002` was canceled
outcome-blind after an unsupported directory-publication operation. Replacement
campaign `barn-icra2027-20260814-003` is running the same 20 sealed CPU tasks
under the amended hard-link publication workflow. No raw log or endpoint
was opened here. Its task is a difficulty stratum but each group shares one
sampled course, so pooled `u_N(E[p_course])` is not generally the exact group
target `E[u_N(p_course)]`. Report it as a preregistered stratum-priority
heuristic unless the sealed post-merge course data justify homogeneity.

The calibration-telemetry DRAFT has static specification/analyzer `GO` but
runtime use remains `HOLD`. Its first clock repair
correctly separates outer cycles from PPO updates, but independent adversarial
audit proved that an all-new or mixed-source cycle can still accompany a
receipt-only claimed update. The new refreeze adds hash-bound per-cycle
pre/post runner counters and branch identity, derives the terminal update total
from that ledger, rejects mixed/new claimed updates and post-target cycles, and
passes 47 local tests. Independent review closed the P1 with 11/11 hostile
clock/source attacks rejected. The final P2 refreeze (`4053c520...` /
`19b07d2f...`) now rejects impossible campaign caps, hardlink aliases,
forged/tampered comparisons, and unreachable replay dispositions; 48 local
tests pass. Final independent re-audit reports no P0/P1/P2, closing the static
specification/analyzer. Execution remains unauthorized until a separate
telemetry overlay/writer/driver/campaign and runtime audit exist. The 100-update
v4 cost design may depend on telemetry only after that runtime closure, or must
freeze an amendment that removes telemetry from its scope.

## Next gates

1. Wait for exact-bundle import/JIT job `9367063` to terminate; inspect only
   scheduler state while it runs, then fetch and checksum-audit its closed
   artifact only after `COMPLETED 0:0`.
2. If and only if that exact import package passes, rerun the bundle's bounded
   one-update gate, then run one two-phase Slurm terminal-chain smoke: one real
   Frontier PPO update plus actual 30-episode evaluation in Phase A, followed
   only after `COMPLETED 0:0` by hardened terminal `sacct` retrieval and atomic
   analyzer-ineligible assembly in Phase B.
3. Repair the four primary v4 remote blockers without mutating protected
   history:
   give R1/R2 an exact compatible core view or audited adapter, canonicalize
   R2 `job-<id>` identity, run all GPU/JAX probes under the byte-closed Python
   3.10.20 environment, and represent unavailable MIG `gpumem` accounting
   honestly while retaining CUDA capacity checks. Then close submission/sacct
   `.batch`/`.extern`, queue-time trust-anchor/TOCTOU, Phase-B environment, and
   real-Hopper schema gates.
4. Build a fresh immutable v4 identity after those changes and obtain an
   independent frozen audit. Only then may its import, one-update, and
   terminal/package engineering gates be staged and run. Neither the audited
   local `d602ce...` bundle nor the hardening snapshot
   `da74eb3e...` is a remote release, and no performance endpoint may be opened
   in this ladder.
5. Implement the calibration runtime as a separately versioned overlay,
   writer, driver, and campaign; then obtain an independent runtime audit
   before enabling telemetry or making Cost-100 depend on it. The frozen
   specification/analyzer itself already has independent `P0=P1=P2=0` GO.
6. Freeze a production-capable v4 cost protocol, then run its production-shape
   100-update pilot to measure post-JIT steady-state throughput, peak memory,
   periodic-evaluation overhead, tie-transform cost, and final-package cost
   without extrapolating the one-update rate.
7. Monitor all 20 BARN tasks by scheduler state only. After every task is
   terminal, run `finalize_barn_ledger.sh` and then
   `finalize_barn_campaign.sh`; do not inspect sealed task outputs before the
   full merge and campaign integrity gate.
8. Keep the original 5090 training gate closed. Treat the isolated
   highest-precision compatibility lane as a separate engineering diagnostic;
   it cannot become benchmark evidence without a complete pre-frozen parity
   receipt and a newly authorized GPU gate.
9. Reproduce group-matched MaxMC and exact Frontier on the frozen development
   seeds before adding the single predeclared Frontier-ACCEL candidate.
10. Run one complete upstream ACCEL cost/reproduction arm before freezing any
   confirmatory array or paper-level comparison.
