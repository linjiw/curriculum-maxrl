# Preregistration — Acrobot U64 over-shooting arm

**Status:** FROZEN on 2026-08-15 before any confirmatory run.
**Schema:** `curriculum-maxrl/acrobot-u64-tournament/v1`
**Lock:** `acrobot_u64/ACROBOT_U64_LOCK.json`

## 1. The question

Every positive score result in the manuscript is confounded with peak hardness.
The score's argmax is

    p*_N = 1 - N^(-1/(N-1))    ->    .5000 (N=2), .1688 (N=16), .0639 (N=64)

so in every existing study the winning arm is also the *harder-peaked* arm. Two
hypotheses make identical predictions in all data collected so far:

- **H_hard**: harder-peaked scores beat p(1-p). (No advance over the
  ProCuRL / SFL / LILO literature — a hand-set difficulty target of ≈.17
  reproduces every number in the paper.)
- **H_peak**: the score should peak where the *deployed* N puts it. (This is
  the paper's actual claim and its only advance.)

The manuscript concedes the gap verbatim at `body_iclr.tex:523-526`: "No arm
pairs a mismatched-N score with a fixed deployed N, so the tournament supports
rollout-aware difficulty targeting rather than peak-location specificity."

**This experiment separates them** by adding an arm that scores with u_64
(p* = .0639) while the deployed estimator stays at N = 16 (p* = .1688). The
u_64 arm deliberately *over-shoots* the deployed peak. If H_peak is true, u_16
beats u_64. If only H_hard is true, u_64 does at least as well as u_16.

## 2. Design

Four arms, 20 paired logical seeds, common random numbers, one fixed task pool.

| arm | sampling score | score peak p* | deployed N |
|---|---|---|---|
| `uniform_shared_h64` | constant 1/8 | — | 16 |
| `p1mp_shared_h64` | p(1-p) = u_2 | .5000 | 16 |
| `u16_shared_h64` | u_16 | .1688 | 16 |
| **`u64_shared_h64`** | **u_64** | **.0639** | **16** |

Everything else is inherited verbatim from the sealed V2 tournament: official
Gymnasium `Acrobot-v1`, the fixed eight-threshold pool, the task-blind H64
actor (640 parameters), the practical dropped-group MaxRL estimator with
N = 16, discounted Beta tracker (decay .7, floor .1, gamma 1.0), learning rate
3e-4, no hindsight, 2,000,000-transition budget, evaluation every 100,000
transitions with 32 episodes, evaluation seed base 1,000,000.

**Only the task-selection score changes across arms.**

### Score/estimator decoupling

`FrontierTeacher.n_rollouts` is used in exactly one place — `utility()` at
`teacher.py:50` — and the engine never reads it. The engine uses its own module
constant `N_ROLLOUTS = 16` for `rollout_group` and for every estimator
decision. `assert_score_estimator_decoupling()` verifies both facts at runtime
and fail-closes, including a static check that the engine source contains no
`.n_rollouts` reference.

### Seeds and RNG

Confirmatory logical seeds are **20000–20019**, identical to V2.
`engine_master_seed(s) = 50_000_000_000 + s * 10_000_000` depends only on the
logical seed, never on the arm index, and all arms share roots as paired common
random numbers. Adding a fourth arm therefore perturbs no existing arm's
stream, and u_64 is CRN-paired with u_16 by construction.

Development seeds 20100–20102 and quick seed 20200 are disjoint from the
confirmatory block and may not enter any inferential result.

## 3. Primary estimands and the frozen decision rule

Metric: **paired target-uniform normalized transition-AUC**, as in V2.

| id | contrast | role |
|---|---|---|
| **A** | u_16 − u_64 | **the new decisive test of H_peak** |
| **B** | u_16 − p(1-p) | **cross-platform replication of the V2 primary** |

Family = {A, B}, Holm-adjusted at familywise .05.

A contrast is **supported** iff both:

1. mean paired difference ≥ **+0.01** (SESOI, inherited from V2); and
2. exact two-sided paired sign-flip p ≤ .05 after Holm adjustment.

With 20 pairs the sign-flip test enumerates 2^20 = 1,048,576 assignments
exactly; no Monte-Carlo substitution is used or permitted.

Secondary, descriptive only, no decision attached: u_64 − p(1-p), and each arm
against uniform.

## 4. Interpretation, fixed in advance

| A (u16−u64) | B (u16−p1mp) | Conclusion |
|---|---|---|
| supported | supported | **H_peak supported.** The deployed-N peak location matters, the confound is broken, and V2 replicates cross-platform. Strongest available outcome. |
| not supported | supported | **H_peak NOT supported.** The replicated finding is "harder-peaked beats p(1-p)"; the paper must say so and drop peak-location specificity as a claim. |
| supported | not supported | Incoherent — u_16 beats u_64 but not p(1-p). Report as inconclusive and investigate; do not claim H_peak. |
| not supported | not supported | **V2 does not replicate on this platform.** Report that first; A is uninterpretable. |

The second row is a **better paper than the status quo**, because it replaces a
silent confound with a measured boundary. This preregistration commits to
reporting it with equal prominence.

## 5. Relationship to the sealed V2 tournament

This is a **derivative**, not an extension. The V2 lock
(`ACROBOT_CURRICULUM_TOURNAMENT_LOCK.json`, sealed 2026-08-08T07:00:25Z)
requires an exact runtime match including `platform` and `machine`, and V2 ran
on macOS arm64. This host is Linux x86_64, so the V2 runner fail-closes here by
design — verified: it refuses `--development` with "runtime mismatch".

Therefore:

- the V2 runner is **not modified and not used** for this campaign;
- all 16 V2-locked source files are vendored byte-identically under
  `acrobot_u64/vendor/` and verified by `verify_vendor_lock.py`
  (**SOURCE LOCK VERIFIED**, all 16 files match);
- the simulation is the vendored engine, called unmodified;
- this campaign carries its own lock pinning this host's runtime and its own
  eight source hashes.

Because the runtime differs, arm B is a genuine **replication test**, not a
recomputation. V2's frozen primary was u_16 − p(1-p) = **+.04803**, CI
[+.02094, +.07385], exact p = .003361.

## 6. Cost

Measured on this host: 200,000 transitions in 10.58 s wall, 52 MB peak RSS,
single core (≈20,800 transitions/s).

- per confirmatory run: 2,000,000 transitions ≈ **2 minutes**
- campaign: 4 arms × 20 seeds = **80 runs** ≈ **2.7 CPU-hours** total
- embarrassingly parallel; each run is independent and writes one JSON

## 7. Execution and stopping rules

1. Development gate: all four arms × seeds 20100–20102 must complete with valid
   runs before any confirmatory run. Development output may not be used
   inferentially.
2. Confirmatory: 80 independent tasks, one per (arm, seed). Each verifies the
   lock before running and writes its output atomically via a `.partial`
   rename, so no truncated file can ever be mistaken for a result.
3. **No inferential quantity is computed until all 80 runs are present and
   valid.** The analyzer refuses an incomplete matrix.
4. The analyzer runs **exactly once**. If it errors, the error is fixed and
   recorded; the decision rule is not revisited.
5. Any arm/seed that fails is re-run from the identical lock and seed. Seeds are
   never substituted, and the seed block is never extended.
6. Gates, SESOI, metric, and the interpretation table above are not changed
   after this document is frozen.

## 8. What this experiment cannot establish

- It does not test any N other than the deployed 16, so it bounds
  peak-location specificity at one operating point, not a general law.
- It uses a 640-parameter policy on one task family; it says nothing about
  scale. MAZE-SCORE is the scale probe and is complementary, not a substitute —
  MAZE-SCORE reproduces the peak-hardness confound (u_32 vs p(1-p)) rather than
  removing it.
- `p1mp` is a common-scaffold score comparator, not an implementation of
  ProCuRL, SFL, PLR, PAIRED, ACCEL, or ALP-GMM.
- A supported A does not show u_16 is optimal; it shows the deployed-N peak
  beats a strictly harder peak at this operating point.
