# Source-locked Acrobot curriculum tournament — V2 amendment

Status: **V2 preregistration amendment**. It becomes effective only when a
matching V2 source/runtime lock is sealed after independent review. No V2
development or confirmatory outcome exists while this status is pending. Quick
smokes are mechanical only and excluded from evidence.

## 1. V1 abort and burned seeds

V1 was stopped by a pre-outcome RNG-domain audit. Its exact original protocol
and lock remain archived as
`ACROBOT_CURRICULUM_TOURNAMENT_PROTOCOL_V1_ABORTED_PRE_OUTCOME_AUDIT.md`
(SHA-256 `c40c15e54aaeba7b5042600ae887167fa3df52765b2db064c77d42cdca773725`)
and `ACROBOT_CURRICULUM_TOURNAMENT_LOCK_V1_ABORTED_PRE_OUTCOME_AUDIT.json`
(SHA-256 `26be1b6abba05bfb47e8f37ac017daa2a1d35964137b129e66c7053fe9907723`).

The V1 logical blocks `19000..19019` and `19100..19102` are permanently
burned. The terminated partial raw artifact is quarantined as
`acrobot_curriculum_tournament_confirmatory_INVALID_ABORTED_rng_domain_overlap.json`
(SHA-256 `7a3afa8b98d3a6cc5290a69562237b7f68312dc8501399d3a8eecd735b232326`).
The untouched V1 development raw and its invalidated gate report are retained
as `acrobot_curriculum_tournament_development_INVALID_ABORTED_rng_domain_overlap.json`
(SHA-256 `3444157dd8ba0376f5c14dbcec0502cee91e4592842cc366f0760df3b93435ed`)
and `acrobot_curriculum_tournament_development_gates_INVALID_ABORTED_rng_domain_overlap.json`
(SHA-256 `e7a7aaaa3a014ce26b5c1464acf98fbb2dabebacb3e8e322c79469b46332ada3`).
Only uniform-arm execution had progressed; no `p(1-p)` or `u16` primary-arm
outcome and no primary contrast was inspected. The partial artifact is retained
for audit only and is ineligible for analysis or pooling.

V1 used adjacent engine seeds, making actor-parameter root `s+1` collide with
the actor-action root of seed `s`. Its ledger also mislabeled the environment
adapter argument `s+1000` as the reset RNG root, although the adapter internally
adds `10003`. V2 repairs both issues rather than amending V1 outcomes.

## 2. Question and claim boundary

This CPU tournament compares target-uniform sampling, a common-scaffold
Thompson-Beta score comparator with utility `p(1-p)`, and exact MaxRL frontier
utility `u16(p)=1-(1-p)^16-p`. The `p(1-p)` arm is not an implementation of the
full ProCuRL, SFL, PLR, ALP, PAIRED, or ACCEL algorithms; only the utility score
changes on the otherwise identical tracker and sampler scaffold.

The primary estimand is target-uniform learning-efficiency AUC for
`u16 - p(1-p)`. This experiment does not test hindsight, architecture,
capacity, transfer causality, or generalization beyond the fixed Acrobot
threshold family.

## 3. Frozen learner and environment

- Official Gymnasium `Acrobot-v1`, including its 500-step limit.
- Ordered thresholds `[-1.5, -1.0, -0.5, 0.0, 0.25, 0.5, 0.7, 1.0]` and the
  existing strict post-transition tip-height verifier.
- One task-blind shared categorical actor with a tanh hidden layer of width 64,
  640 trainable parameters, learning rate `3e-4`, and plain SGD ascent.
- Practical dropped-group MaxRL weights
  `1{K>0}(r_i/K - 1/16)`, with no hindsight or relabel-derived update.
- Complete groups of `N=16` rollouts. A group is never cut at a transition
  boundary.
- Both adaptive arms use the same per-task discounted Beta tracker, Thompson
  draws, decay `0.7`, uniform floor `0.1`, and exponent `gamma=1`; only utility
  differs. Uniform requests every threshold with probability `1/8`.

The wrapper imports the existing neural Acrobot engine, temporarily replaces
its teacher factory inside a sequential process-local context, and restores it
in `finally`. It retains every full task-probability vector, posterior mean,
group record, transition checkpoint, sampled-group checkpoint, and optimizer-
update checkpoint.

## 4. Logical seeds and globally disjoint RNG roots

V2 uses fresh logical blocks: confirmation `20000..20019`, development
`20100..20102`, and quick smoke `20200`. For logical seed `s`, the engine master
seed is

`m(s) = 50,000,000,000 + 10,000,000*s`.

The exact RNG roots within a logical pair are:

| Domain | Root |
|---|---:|
| actor parameters | `m` |
| actor actions | `m+1` |
| teacher | `m+10000` |
| training reset-seed generator | `m+11003` |
| evaluation episode-seed generator | `m+1000000` |
| evaluation action-seed generator | `m+1000001` |

The engine passes `m+1000` as the adapter seed argument; the adapter creates
its training reset-seed generator at `(m+1000)+10003=m+11003`. The 10,000,000
stride exceeds every domain offset. The collision audit requires global
uniqueness over every registered `(logical seed, domain)` pair and no overlap
with any prior, reserved, quick, development, or burned logical block. The
three arms intentionally reuse one logical pair's exact roots as paired common
random numbers; this within-pair reuse is not a collision.

Raw runs retain the logical seed as `seed` and separately record the master,
adapter argument, and all six roots. The independent analyzer derives them
again without importing the runner and fails on any mismatch.

## 5. Development and launch gate

The registered development run uses all three arms and logical seeds
`20100..20102`, a 200,000-transition crossing target per run, evaluations at
initialization and at the first complete group crossing every 50,000
transitions, and 16 shared nested evaluation trajectories per checkpoint.
The final group is completed, so an individual run can exceed the target by
at most one 16-by-500-step group (8,000 transitions).
Development is used only for these outcome-blind gates:

1. every run passes numerical, accounting, parameter-count, verifier,
   checkpoint-cadence, curve-recomputation, and evaluation-state checks;
2. every task is visited at least once when pooled across arms and seeds;
3. each adaptive arm exhibits a nonuniform requested-task distribution, with
   total variation recomputed from saved probability vectors;
4. pooled groups include dead, mixed, and all-pass regimes; and
5. native-success checkpoint values exhibit variation.

No contrast, effect direction, p-value, interval, or minimum effect enters the
launch gate. Development raw data must be retained in the repository beside
the gate report. The full runner verifies the gate's exact required keys and
policy, the source-lock hash, the raw artifact's project-relative path and
SHA-256, and the raw artifact's locked provenance before accepting it.
Evidence-bearing runs accept only the canonical repository lock at
`frontier_rl/examples/ACROBOT_CURRICULUM_TOURNAMENT_LOCK.json`; a copied or
alternate lock path is rejected even if its bytes are identical.

`--quick` uses only logical seed `20200`, 8,000 transitions, a 4,000-transition
evaluation interval, and two shared nested evaluation trajectories. Quick
also completes its final group and can exceed its crossing target by at most
8,000 transitions. Quick
analysis validates raw ledgers without reading or requiring a source lock and
cannot create a launch gate.

## 6. Sealed confirmation

Run exactly 60 complete runs: three arms by 20 logical seeds `20000..20019`.
Each run receives 2,000,000 actual environment transitions; the final group is
completed and can overshoot by at most one 16-by-500-step group.

Evaluate at initialization and at the first complete group crossing each
100,000-transition boundary. Because the policy is shared and thresholds are
nested, each checkpoint uses 32 shared trajectories total, not eight
independent per-task batches: each trajectory's maximum height is reused to
score all eight thresholds. Native success must equal task-7 pass rate. Fixed
episode and action roots give common random numbers across arms and checkpoints
within a logical pair. Evaluation must preserve training parameters, counters,
and RNG states. The retained checkpoint records contain aggregate nested
success curves and the registered evaluation grid, but not a trajectory-level
ledger of individual evaluation episodes. The analyzer therefore verifies the
evaluation count and grid from those aggregates rather than replaying each
evaluation trajectory.

Every run, including an invalid run, is retained. There is no seed replacement,
interim analysis, outcome-dependent stopping, or tuning. Before development,
before every sealed run, and after finalization, the runner must match the exact
V2 source lock under CPython `3.12.13`, NumPy `2.5.1`, and Gymnasium `1.3.0`.

## 7. Registered primary analysis and decision

For logical seed `s` and arm `m`, let `A(s,m)` be normalized trapezoidal area
under the target-uniform mean pass-rate curve against actual environment
transitions, from initialization through the complete terminal group. The sole
primary contrast is `d_s=A(s,u16)-A(s,p(1-p))`.

The analyzer reports the mean of 20 paired differences, a 20,000-resample
paired-seed percentile-bootstrap 95% interval for estimation support, and the
exact two-sided sign-flip p-value over all `2^20` assignments. The randomization
interpretation requires paired effects to be sign-exchangeable under the sharp
null; pairing and common random numbers do not make that assumption automatic.

Before any V2 primary-arm outcome, the practical SESOI is fixed at `+0.01`
normalized transition-AUC: one average percentage point over the paid
transition trajectory. On a 2,000,000-transition paid horizon, `+0.01`
corresponds to 20,000 pass-rate-by-transition units. Each checkpoint's
target-uniform mean pass rate has resolution `1/(32*8)=0.00390625`, so the
SESOI is about 2.56 such resolution units. This is a judgment-based practical
convention fixed before any V2 primary-arm outcome and was not derived from V1
or aborted primary outcomes. Efficacy is supported if and only if the mean
paired `u16-p(1-p)` difference is at least `+0.01` **and** the exact two-sided
`p<=0.05`; the direction must therefore be positive. Otherwise the result is
"not confirmed," never equivalence. The bootstrap interval does not create a
separate decision rule. There is one primary test and no primary multiplicity
adjustment. The p-value tests a zero paired contrast, not the `+0.01` SESOI;
clearing both filters does not establish statistically that the population
effect is at least `+0.01`.

The earlier descriptive V3 `u16-uniform` result was known when this amendment
was written. The primary `u16-p(1-p)` contrast had not been run or inspected.

Two secondary target-uniform transition-AUC tests compare `p(1-p)-uniform` and
`u16-uniform`; their exact p-values form one Holm step-down family at FWER .05
and cannot rescue the primary. A secondary arm is called supported only when
its mean contrast is positive and its Holm step-down rejection is true; a
significant negative contrast is never labeled efficacy.

## 8. Registered descriptive diagnostics

Native-success AUC, native-return AUC, final native success/return, realized
practical-MaxRL mass per group and per million transitions, and nonzero-mass
group fraction are descriptive. Target-uniform AUC on sampled-group and
optimizer-update axes, plus sampled groups and optimizer updates per million
transitions, are also predeclared descriptive cost-composition diagnostics:
threshold-dependent stopping horizons can make transition-AUC differ because
samplers purchase different group/update mixtures. These endpoints carry no
confirmatory decision and cannot rescue the primary.

The analyzer reconstructs transition, sampled-group, and optimizer-update axes
from raw group/checkpoint ledgers; recomputes probability TV, mean/hardest/native
curves, final fields, and every saved AUC/rate; checks shapes, bounds, nesting,
CRN roots, and exact checkpoint crossings; and fails closed on any mismatch.
If any registered run is missing or invalid, confirmatory inference is not
performed. Raw execution and independent analysis remain separate artifacts.
