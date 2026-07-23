# Acrobot neural hindsight V5 protocol

Status: **registration-ready design**.  No registered V5 seed may be touched
before a companion source/runtime lock is created after runner, independent
analyzer, protocol, and test review.  The companion lock and resulting
artifacts, when present, are authoritative for seal/execution times; this
protocol does not claim that a lock or experiment artifact already exists.

## 1. Why V5 exists

V4A is immutable failed calibration evidence.  Its scale-zero-only development
grid could validate preview mechanics, but it could not establish that natural
hindsight relabels occur and produce finite applied updates in the positive
scale cells that V4B would have tested.  V5 does not revise, overwrite, pool,
or silently reinterpret any V4 artifact.  It starts with fresh seed blocks and
runs the complete `3 learning rates x 3 hindsight scales` grid during
feasibility.

V5 asks two separate questions:

1. **Can this exact implementation support a fair update-matched factorial?**
   V5A answers this using source/runtime equality, first-exact-update prefixes,
   optimizer-source mechanics, natural relabel coverage, teacher nonuniformity,
   and a serial runtime projection.  Evaluation-performance fields and scale
   contrasts are excluded from the launch rule.
2. **Does verifier-valid hindsight improve update-indexed learning, and is the
   response distinguishable from a restricted learning-rate/scale model?**
   V5B answers this on a fresh 20-seed confirmatory block, if and only if every
   V5A gate passes and an independent verifier authorizes launch.

Because positive hindsight changes the policy path, V5A is not called
"treatment blind" or "effect blind."  Its defensible property is narrower:
the launch rule is **learning-outcome-field blind**.  Natural relabel frequency,
update-source counters, and group regimes are post-treatment mechanics and are
used only as feasibility/relevance checks.

## 2. Frozen environment and algorithm

- Gymnasium environment: unmodified `Acrobot-v1` dynamics and native 500-step
  time limit.
- Curriculum tasks: eight nested strict post-transition predicates

  `tip_height(observation_after) > h`,

  with `h in [-1.5, -1.0, -0.5, 0.0, 0.25, 0.5, 0.7, 1.0]`.
- Policy: one task-agnostic shared `6 -> tanh(64) -> 3` categorical actor,
  without output bias: 640 total and active parameters.
- Group size: `N=16` complete rollouts.
- Teacher: frontier utility `u_16(p)=1-(1-p)^16-p`, discounted Beta evidence
  with `gamma=1`, decay `0.7`, and uniform floor `0.1`.
- Teacher evidence: the requested task and its original 16 binary outcomes
  only.  A hindsight relabel never supplies extra evidence to the teacher.
- Live update: centered MaxRL group weights on a requested mixed group.
- Hindsight eligibility: the requested group is dead (`K=0`), the policy is
  shared, and a lower threshold has a verified mixed outcome.
- Hindsight relabel: choose the hardest lower mixed predicate, recompute every
  binary reward from stored post-transition observations, and truncate each
  successful trajectory at its first strict threshold crossing.  The credited
  task must be strictly lower than the requested task.
- Optimizer: plain SGD ascent.  Score terms are summed over all retained
  transitions at the frozen group parameters; there is no clipping, trajectory
  normalization, momentum, or adaptive optimizer state.
- Matching axis: nonzero applied optimizer updates.  A zero-gradient attempt is
  not an update.

For an identical frozen auxiliary group with score gradient `g`, learning-rate
multiplier `a`, hindsight scale `s`, and base rate `eta=3e-4`, the mechanical
contract is

`Delta theta_aux(a,s) = eta * a * s * g`.

This identity does not claim that full learning curves at equal `a*s` must be
equal.  Live updates, natural relabel timing/source composition, and policy
trajectories can differ across cells.

## 3. Reused implementation provenance

The V5 orchestration has new V5 schemas, schedules, gates, authorization, and
analysis.  It reuses only the serial `_instrumented_run` execution engine in
`run_acrobot_hindsight_v4.py`, which in turn calls the frozen neural Acrobot
loop.  V5 locks the defining V4 module, the neural loop, adapter, teacher,
estimators, interfaces, eager package imports, and all named tests by exact
SHA-256.  Every artifact records those hashes and explicitly records that no
V4 artifact was reused.

Before sealing or running, an executable engine-contract check requires:

- instrumentation checkpoints exactly `{250,400}`;
- evaluation cadence exactly 50 applied updates;
- transition group-start cap 4,000,000 and maximum complete-group overshoot
  8,000;
- the same imported neural-loop module object;
- exactly 16 rollouts and the exact eight thresholds;
- teacher `(gamma,decay,floor)=(1,0.7,0.1)`;
- Acrobot maximum episode length 500; and
- shared-H64 parameter counts `(total,active)=(640,640)`.

The V4 instrumentation temporarily patches module globals and is therefore
serial-only.  V5 must not execute registered cells in threads.

## 4. Seed provenance and collision audit

V5A uses fresh seeds `15000..15002`; V5B uses fresh seeds `16000..16019`.
Before a lock is written or used, the runner checks both blocks against the
explicit V1--V4 Acrobot seed ledger and against each other.  The source and
protocol files carrying this ledger are hash-locked.  A collision is a hard
stop.

The deterministic preview shadow uses already-executed exploratory smoke seed
`100`, intentionally outside the fresh V5 blocks.  The nine-cell algebraic
fixture uses synthetic actor seed `9907`; it performs no Gymnasium rollout and
does not touch a registered seed.

## 5. Deterministic pre-seed mechanical fixtures

Both fixtures are saved atomically before the first registered V5A run and are
recomputed exactly on resume.

### 5.1 Nine-cell scale fixture

Nine identical actors receive one identical synthetic frozen auxiliary group,
one actor for every `a in {0.5,1,2}` and `s in {0,1,2}`.

- Every positive-scale cell must satisfy the full vector equality
  `Delta theta = 3e-4*a*s*g` within fixed numerical tolerance, increment the
  update/applied counters exactly once, preserve action RNG state, and produce
  a strictly positive finite parameter-step norm.
- Every scale-zero cell must produce a strictly positive finite unscaled
  gradient preview while leaving parameters, counters, and action RNG exactly
  unchanged.
- Credited/source metadata must represent a strictly lower relabeled task.
- `||Delta theta||/(a*s)` must be common across all six positive cells.

This fixture forces mechanics only.  It does not force an event in any
registered V5A run and cannot satisfy the natural-relevance gates below.

### 5.2 Preview/no-hindsight shadow

At learning-rate multiplier 1 and scale zero, the scale-stage preview path and
an otherwise identical no-hindsight path must have identical training-group
trace hashes, final actor training-state hashes, resource coordinates,
mechanical counters, update diagnostics, and group diagnostics.  At least one
eligible positive finite nonmutating preview must be exercised.  Evaluation
performance/return/entropy arrays are not part of this gate-relevant equality.

## 6. V5A: full-grid natural feasibility

Run all nine cells on paired seeds `15000,15001,15002`:

| learning-rate multiplier `a` | hindsight scale `s` |
|---:|---:|
| 0.5 | 0, 1, 2 |
| 1.0 | 0, 1, 2 |
| 2.0 | 0, 1, 2 |

The base learning rate is `3e-4`.  Evaluate at update zero and every 50 nonzero
applied updates with 32 episodes per task and fixed per-seed common random
numbers.  No group starts once transitions are at least 4,000,000; a started
16-rollout group completes, permitting at most 8,000 transitions of overshoot.

### 6.1 Fresh update budget

Select the confirmatory budget from all 27 V5A runs and no earlier artifact:

1. `U*=400` if every run reaches 400 nonzero applied updates;
2. otherwise `U*=250` if every run reaches 250;
3. otherwise stop and do not authorize V5B.

Every launch gate below is computed on each run's groups/updates through the
**first exact `U*` crossing only**.  If 250 is selected, saved diagnostics and
outcomes after that crossing cannot veto or rescue launch.  A process failure
that prevents the runner from returning and retaining an auditable exact
prefix still invalidates that run; V5 does not claim mid-run checkpoint
recovery.  Full artifacts may still be audited descriptively outside the
launch decision.

### 6.2 All launch gates

V5B is authorized only if all seven gates pass:

1. **Lock, engine, fixture, schedule, and prefix invariants.** Exact live
   source/runtime/engine/seed-audit equality; exact 9-cell/3-seed order and
   configs; deterministic nine-cell fixture passes; every selected prefix has
   contiguous complete-group transition accounting, strict verifier/task
   accounting, at most one update per group, consecutive positive finite
   applied updates, valid source records, exact cap semantics, and evaluation
   state preservation.
2. **Fresh all-27 first-exact prefix.** `U*` follows the rule above and every
   run has an exact first-crossing prefix with evaluation resource coordinates
   `[0,50,...,U*]`.
3. **Scale-zero preview and shadow.** In every scale-zero prefix, eligible
   candidate group ids equal preview group ids; every preview is positive,
   finite, frozen-parameter, and nonmutating; requested-live updates equal
   `U*`; applied hindsight updates equal zero; and the deterministic shadow
   passes.
4. **Positive-scale application and natural relevance.** In every positive
   prefix, every encountered eligible candidate maps one-to-one to a finite,
   strictly positive applied `hindsight_relabel` update with a strictly lower
   credited task and no preview-only record.  Additionally, each positive
   cell has at least one such applied natural update pooled across its three
   seeds.  Each scale-zero cell has at least one eligible valid preview
   candidate pooled across its three seeds.
5. **Natural regimes.** Each of the nine cells, pooled across its three seeds,
   observes at least one dead, mixed, and all-pass requested group.
6. **Teacher activity.** In every run, among selected-prefix groups whose
   transition start is at least 200,000, mean total-variation distance from
   uniform is strictly greater than 0.05.
7. **Serial runtime.** With `w_j(U*)` the exact wall seconds immediately after
   run `j` reaches `U*`,

   `180 * max_j w_j(U*) / 3600 <= 18 hours`.

No launch gate reads pass-rate curves, returns, entropy curves, AUCs, final
performance, learning-rate/scale performance contrasts, or any V4 outcome.
The independent analyzer constructs and hashes a technical gate projection
that contains none of those field names and independently recomputes the
selected-prefix decision.

## 7. V5B: confirmatory 3 x 3 factorial

Only a post-V5A amendment with explicit authorization and an independent
passing V5A report may create the V5B source lock.  V5B uses paired seeds
`16000..16019`, all nine cells, and the selected `U*`.  This is exactly 180
runs.  Every run must finish uncensored at exactly `U*` and have evaluation
updates exactly `[0,50,...,U*]`.

The scale-specific relabel contract is also a V5B validity rule.  In a
scale-zero run, every eligible candidate must have one positive finite
nonmutating preview and there may be no applied hindsight update.  In a
positive-scale run, eligible candidates must partition exactly into (i)
positive finite applied hindsight updates and (ii) explicitly recorded,
finite zero-gradient hindsight attempts with zero gradient and update norms;
the two sets must be disjoint and every record must carry exact dead-source
group and strictly-lower credited-task metadata.  A zero-gradient attempt is
not counted on the update matching axis, but it is permitted when this audit
trail is complete.  An unaccounted or malformed candidate invalidates that run
and therefore the all-or-nothing family.

The primary metric for every cell/seed is normalized target-uniform mean-pass
AUC indexed by nonzero optimizer updates, including update zero:

`AUC = (1/U*) integral_0^U* mean_pass(u) du`,

using the registered trapezoidal checkpoints.  Transition counts and
source-wise cumulative step norms are diagnostics, not matching axes.

### 7.1 Four-contrast family

Let `Y(a,s)` denote one paired seed's update-indexed AUC.

- `C1 = Y(1,1)-Y(1,0)`.
- `C2 = Y(1,2)-Y(1,1)`.
- `C3 = [Y(0.5,2)-Y(0.5,0)]-[Y(1,1)-Y(1,0)]`.
- `C4 = [Y(1,2)-Y(1,0)]-[Y(2,1)-Y(2,0)]`.

For each contrast, enumerate all `2^20` paired sign assignments and compute the
exact two-sided sign-flip p-value.  Apply Holm step-down correction to the four
raw p-values at familywise alpha 0.05.  Report a descriptive 20,000-resample
paired-seed bootstrap interval using fixed contrast seeds `55000..55003`.
The randomization interpretation assumes independent seed-level pairs and
sign-exchangeable null contrasts.  Exhaustive `2^20` enumeration makes the
calculation exact conditional on that assumption; it does not establish the
assumption automatically.

- Directional C1/C2 support requires mean contrast at least `+0.03` **and**
  Holm rejection.
- Material C3/C4 restricted-separability departure requires absolute mean
  contrast at least `0.03` **and** Holm rejection.

The family is all-or-nothing: if any lock, source, config, seed, numeric,
accounting, cap, terminal-coordinate, or evaluation-coordinate check fails in
any of the 180 runs, no member of the primary family is computed or claimed.
There is no secondary rescue analysis.

## 8. Interpretation boundary

C1/C2 can support local update-matched performance effects of the complete
verifier-valid hindsight procedure at the base learning rate.  They do not
establish transition efficiency, wall-clock efficiency, transfer to another
environment, or an unbiased hindsight estimator.

C3/C4 test departure from the restricted representation
`Y(a,s)=F(a)+G(a*s)`.  Such departure is useful evidence that the observed
factorial is not described solely by a main learning-rate term plus the product
`a*s`.  It does not separately identify "semantic data value": natural relabel
frequency, source composition, trajectory truncation, and the policy path also
change with treatment.

## 9. Operational sequence

All commands use module entry points so project imports work from a clean
checkout.

1. Review V5 source/protocol/tests; run the declared preflight suite.
2. Seal V5A once.  This writes hashes only and touches no registered seed.
3. Run/resume V5A.  Artifacts are atomic and append only complete seed records.
4. Run the independent V5 analyzer on V5A.
5. Inspect only the saved technical gates and independent authorization result.
6. If every gate passes, create the explicit V5B amendment/source lock.
7. Run/resume all 180 V5B runs.
8. Run the independent analyzer; accept the primary family only if runner and
   analyzer agree exactly.

Existing locks, amendments, reports, and terminal artifacts are never
overwritten.  Any source/runtime/schedule/hash mismatch fails before a new
registered seed is touched.
