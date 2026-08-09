# Digits exact-probability estimator × sampler factorial

Status: **frozen before learning-rate development or confirmation, pending an
independent pre-seal review and a separate execution authorization**. No public
pre-execution commit exists. Engineering seed `33000` is non-evidentiary and
may be run only with a truncated schedule. Development seeds `31000..31003`
and confirmation seeds `32000..32023` must not execute until the authorization
file bound to this source lock permits the relevant phase.

## Question and claim boundary

This controlled contextual-bandit study asks whether the preferred
cross-example curriculum depends on the within-example policy-gradient
estimator. It crosses the practical dropped-group MaxRL estimator and RLOO
with target-uniform, `p(1-p)`, and `u_8(p)=1-(1-p)^8-p` sampling. It uses a
real classification dataset with an exact policy success probability; it is
not trajectory RL, GRPO, ImageNet, or a faithful implementation of native
ProCuRL.

The registered prediction is an interaction: practical MaxRL should favor
`u8` over `p1mp`, while RLOO should favor `p1mp` over `u8`.

## Frozen data and runtime

- Runtime: CPython 3.11.14, NumPy 1.26.4, SciPy 1.13.1, PyTorch 2.8.0,
  scikit-learn 1.5.2; evidentiary execution is CPU-only.
- Torch intra-op/inter-op and OMP, MKL, OpenBLAS, VecLib, and NumExpr thread
  counts are fixed to one and recorded in every run. Deterministic Torch
  algorithms are required. Serial and process-parallel zero-rate engineering
  artifacts must be byte-identical except for explicitly unbound timing files.
  Those sidecars record direct/serial/process-pool mode and the requested
  worker count; the engineering audit requires one-worker serial and at least
  two-worker process execution.
- Dataset: `sklearn.datasets.load_digits`; 64 pixels are divided by 16 and
  represented as float64.
- Sealed test: `StratifiedShuffleSplit(n_splits=1,test_size=360,
  random_state=20260808)` on all 1,797 rows.
- Development: the same splitter with `test_size=360` and
  `random_state=20260809` on the remaining 1,437 rows. The result is 1,077
  train, 360 development, and 360 test rows.
- Original-dataset indices, class counts, and NPY-byte hashes of raw data,
  target, normalized data, and every index vector are stored in
  `digits_split_manifest.json`. The runner loads these stored indices and
  never regenerates a split.

## Frozen learner and six cells

The policy is `Linear(64,64) -> ReLU -> Linear(64,10)` with biases, exactly
4,810 trainable parameters. It uses float64 SGD with momentum 0.9. There is no
dropout, batch normalization, augmentation, entropy bonus, critic, clipping,
or weight decay. Per logical seed, all six cells begin from the identical
checkpoint.

For each selected example, eight actions are sampled and receive binary
correct-class rewards. If `K` actions are correct, action coefficients are

```text
practical MaxRL:  1{K>0} (r_j/K - 1/8)
RLOO:              (1/8) (r_j - (K-r_j)/7).
```

Every update selects 64 examples with replacement and minimizes

```text
-(1/64) sum_i sum_j stopgrad(w_ij) log pi(a_ij | x_i).
```

At the start of every step, every arm performs a no-gradient float64 forward
pass over the complete training pool and obtains the exact correct-class
probability `p_i`. Scores are `1`, `p_i(1-p_i)`, or
`1-(1-p_i)^8-p_i`. Sampling probabilities are

```text
q_i = 0.1/1077 + 0.9 s_i / sum_l s_l.
```

The mathematically unreachable non-finite/non-positive score-sum fallback is
uniform. No posterior, probe, interpolation, or stale estimate is used.

## Budget, common random numbers, and evaluation

There are exactly 512 updates, 64 groups per update, and 8 actions per group:
32,768 groups and 262,144 paid actions. Full-pool scoring therefore evaluates
551,424 training examples in every cell. Evaluation is cost-free and occurs
at action budget zero, every 8,192 actions, and at the terminal budget.

For each seed, NumPy `PCG64DXSM` domain-separated tapes contain all
`512x64` inverse-CDF task uniforms and `512x64x8` action uniforms. The same
complete tapes and initialization are reused in all six cells. Current `q`
and current categorical policy probabilities transform the tapes; sampled
examples and actions are consequently paired but need not remain identical.

For any evaluation dataset `D`, exact coverage is

```text
C_k(D) = mean_i [1 - (1-p_i)^k],  k in {1,2,4,8,16,32}.
```

The primary outcome `Y` is normalized trapezoidal AUC of sealed-test `C_8`
against paid action budget. Secondary curves include mean correct-class
probability, all registered `C_k`, NLL, multiclass Brier score, top-1 and
macro-class metrics, dead/mixed/all-pass group fractions, coefficient mass,
gradient norm, and decile-binned sampler exposure.

## Learning-rate development without test access

Run every six-cell combination at learning rates `{0.03,0.1,0.3,1,3}` for
each of seeds `31000..31003`. Development artifacts omit sealed-test arrays
and metrics. For each estimator separately, select the rate maximizing the
equal average development `C8` AUC across its three samplers and four paired
blocks; an exact tie chooses the smaller rate. Select a common rate analogously
by averaging all six cells and four blocks. Valid selection requires every
registered run to be complete and finite.

The development launch gates additionally require stored split integrity;
formula/mass audits; the `N=2` MaxRL/RLOO and `u2/p1mp` identities; a zero-rate
common-random-number dry run; exact budgets, checkpoints, initialization and
tape hashes; one valid selected rate per estimator; and median development
`C8` improvement of at least 0.02 across the two selected-rate uniform arms.
The separate development authorization must bind the SHA-256 of a passing
serial-versus-parallel zero-rate engineering audit under the same source lock.

Only after rates are frozen may test outcomes be generated. Test outcomes
must never influence learning-rate selection.

## Confirmation and registered inference

Run the six cells at their estimator-specific selected rates for all 24 fresh
seeds `32000..32023`. Also run all six cells at the frozen common rate as a
registered optimizer-sensitivity analysis. For seed `s`, define

```text
I_s = 0.5 * [(Y_MaxRL,u8 + Y_RLOO,p1mp)
             - (Y_MaxRL,p1mp + Y_RLOO,u8)].
```

The primary is supported if and only if mean `I >= 0.01`, the fixed
100,000-resample paired percentile-bootstrap 95% interval has lower endpoint
above zero, the exact two-sided `2^24` sign-flip p-value is at most 0.05, and
the treatment-delivery gate below passes. Bootstrap RNG is PCG64DXSM seed
20260808.

Four predeclared two-sided paired simple-effect tests form one Holm family:
MaxRL `u8-p1mp`, RLOO `p1mp-u8`, MaxRL `u8-uniform`, and RLOO
`p1mp-uniform`. “Both matched” is allowed only when each estimator's matched
simple effect has positive mean, positive bootstrap lower endpoint, and a
Holm-rejected p-value. Uniform anchors are secondary and cannot rescue the
primary.

Treatment delivery requires the action-budget-weighted mean total variation
between `q_u8` and `q_p1mp` to be at least 0.02. A complete paired block is all
six cells for one seed. A failed cell removes the entire block with no
replacement; fewer than 20 complete confirmation blocks makes the result
inconclusive. A positive tuned-rate primary accompanied by a sign reversal at
the common rate must be labeled optimizer-sensitive. Tuned/common sensitivity
is computed only on their intersection of complete blocks and is inconclusive
when that intersection has fewer than 20 blocks.

Confirmation requires a separate authorization, created only after
development analysis, that binds the exact SHA-256 of the frozen, fully
passing learning-rate-selection artifact. An authorization cannot permit both
development and confirmation.

## Artifact and failure standard

Each completed run stores source/runtime/data/split hashes; the full selected
example and action/reward ledger; every step's full training `p` and `q`
vectors and selected categorical probabilities; all coefficient weights and
masses; exact paid/scoring/update/evaluation counts; full per-example
train/dev/test categorical probabilities at evaluation checkpoints (test is
omitted during development); five model/optimizer/RNG recovery checkpoints at
steps 0, 128, 256, 384, and 512; sampler exposure; wall time; and a failure
ledger. Recovery state includes Torch and Python RNG state plus the terminal
states and hashes of both NumPy tapes. Wall time and scheduler/worker
provenance are retained in an explicitly unbound timing sidecar so scientific
artifacts remain scheduler-independent. Checkpoints use an exact,
tensor-safe schema and are loaded with `weights_only=True`. The independent
analyzer deterministically replays initialization and every learner/SGD step,
including full-pool `p`, `q`, selected examples, categorical probabilities,
actions, rewards, coefficients, loss, gradient norm, evaluations, model and
optimizer checkpoints, RNG state, metrics, budgets, AUCs, gates, and
registered inference. Summary, NPZ, checkpoint, and directory schemas fail
closed on duplicates, extras, wrong types/dtypes/shapes, non-finite values,
path traversal, or disagreement between artifact identity and its
learning-rate/seed/cell directory labels. A confirmation failure directory is
either empty/missing or contains exactly one schema-validated `failure.json`.
Development contains no sealed-test outcome field or array.

MPS may be used only for explicitly labeled timing work. It cannot enter the
development or confirmatory evidence.

## Invocation

Run commands from the repository root and select the nested locked project:

```bash
uv run --project curriculum_maxrl/digits_factorial --locked \
  python -m curriculum_maxrl.digits_factorial.verify_portable --source-only
uv run --project curriculum_maxrl/digits_factorial --locked \
  python -m pytest curriculum_maxrl/digits_factorial/tests -q
```

Evidence-bearing phases additionally require a separately created execution
authorization bound to the reviewed `SOURCE_LOCK.json`; the runner fails
closed without it.
