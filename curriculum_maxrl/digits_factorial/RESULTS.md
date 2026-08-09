# Digits exact-probability factorial: registered negative result

Updated 2026-08-08. This document reports the source-locked
estimator-by-sampler factorial defined in `PROTOCOL.md`. The primary
estimator-by-sampler interaction was **not supported**. The result is useful
because it separates the exact coefficient-mass identity from a much stronger,
and false here, claim that coefficient mass universally identifies the best
curriculum.

## Claim boundary

This is a controlled contextual bandit on `sklearn.datasets.load_digits`, not
trajectory RL, GRPO, ImageNet, language-model RLVR, or native ProCuRL. The
policy is a 4,810-parameter float64 MLP. Its correct-class probability is
available exactly and is recomputed over the full training pool before every
update. The experiment crosses practical dropped-group MaxRL and RLOO with
uniform, `p(1-p)`, and
`u_8(p)=1-(1-p)^8-p` example sampling. Consequently, this experiment can test
the registered estimator-by-sampler prediction cleanly; it cannot establish a
general ranking of curriculum algorithms or neural-RL environments.

## Frozen design and budgets

- Data: 1,077 train, 360 development, and 360 sealed-test examples from a
  stored stratified split.
- Per run: 512 updates, 64 sampled examples per update, and eight categorical
  actions per example, for 32,768 groups and 262,144 paid actions. Full-pool
  scoring evaluates 551,424 training examples per run. Test coverage is
  evaluated exactly.
- Development: five learning rates by four seeds by six cells, or 120 runs
  and 31,457,280 paid actions. Every registered gate passed. The selected
  MaxRL, RLOO, and common rates were all exactly `0.1`.
- Confirmation: 24 fresh paired seed blocks and six cells, or 144 runs and
  37,748,736 paid actions for each of the estimator-selected and common-rate
  schedules. All blocks completed with no failures.
- The two confirmation schedules therefore contain 288 executed runs and
  75,497,472 paid actions. Because all selected rates equal the common rate,
  every paired ledger and recovery checkpoint is byte-identical, and summary
  content is identical after removing the required phase/authorization
  labels. The second schedule is an identity check, **not an independent
  replication or optimizer-sensitivity experiment**.
- Across development and both confirmation directories, 408 runs executed
  106,954,752 paid actions. Statistical inference nevertheless uses the 24
  fresh paired confirmation blocks, not run count or action count as the
  sample size.

The engineering audit enumerated all 256 binary reward vectors at `N=8`,
checked 512 estimator vectors, verified the `N=2` estimator and sampler
identities, and obtained byte-identical scientific files from serial and
two-worker execution. Development artifacts contained no sealed-test
outcomes.

## Registered result

The primary per-block interaction was

```text
I = 0.5 * [(Y_MaxRL,u8 + Y_RLOO,p1mp)
           - (Y_MaxRL,p1mp + Y_RLOO,u8)],
```

where `Y` is normalized sealed-test `C8` AUC against paid action budget. The
point estimate, `+0.0158852`, exceeded the registered `+0.01` threshold, and
treatment delivery passed (`TV(q_u8,q_p1mp)=0.570475 >= 0.02`). But the fixed
100,000-resample paired bootstrap interval was
`[-0.0168606, +0.0471202]`, the exact two-sided `2^24` sign-flip
`p=0.3495574`, and only 15/24 blocks were positive. The primary therefore
failed its conjunctive decision rule.

Mean sealed-test `C8` AUCs were:

| estimator | uniform | `p(1-p)` | `u8` |
|---|---:|---:|---:|
| practical MaxRL | 0.960836 | 0.639630 | 0.848050 |
| RLOO | 0.817216 | 0.441408 | 0.618058 |

The four predeclared simple effects and uniform anchors say exactly where the
prediction failed:

| registered contrast | mean | paired bootstrap 95% interval | exact two-sided sign-flip `p` | signs | interpretation |
|---|---:|---:|---:|---:|---|
| MaxRL: `u8-p1mp` | +0.208421 | [+0.167905, +0.247435] | 2.3842e-7 | 23/24 positive | Holm-supported in the predicted direction |
| RLOO: `p1mp-u8` | -0.176650 | [-0.220419, -0.135926] | 1.1921e-7 | 0/24 positive | Holm-rejected in the direction opposite the prediction |
| MaxRL: `u8-uniform` | -0.112786 | [-0.139887, -0.086643] | 1.1921e-7 | 0/24 positive | matched sampler below uniform |
| RLOO: `p1mp-uniform` | -0.375808 | [-0.417757, -0.339115] | 1.1921e-7 | 0/24 positive | matched sampler below uniform |

Thus `u8` decisively beats `p(1-p)` under practical MaxRL in this controlled
setting, but RLOO also favors `u8`, reversing its registered prediction, and
both estimator-matched samplers lose to uniform. The result does not challenge
the algebraic coefficient-mass identity. It rejects the stronger empirical
story that matching a sampler to expected coefficient mass is sufficient for
curriculum optimality. Coefficient mass measures estimator activity; it omits
score-gradient direction, gradient norm and variance, example diversity,
optimizer dynamics, parameter sharing, and transfer.

## Integrity chain

The following are full SHA-256 file checksums. Timing sidecars are explicitly
unbound metadata.

| artifact | SHA-256 |
|---|---|
| `SOURCE_LOCK.json` | `d72b93a29a2e6a096a6acb0611f69fe6df9dcda80256000aed6de5208ef4eb36` |
| `digits_split_manifest.json` | `13dbc30cc5143edb043d76d76aac18bcc3a456b174a18ba488498fb99eab5e3f` |
| `engineering/reseal_v3/engineering_audit.json` | `d448a22793dc5a52ae7809dfc7572e3d669bebc5f8ef8d00a9865a0bb16150d1` |
| `engineering/reseal_v3/independent_preseal_review.json` | `0c387ddd1b2bb49d2be6e1eacff96c9281cfa17eea85f01b425733c9aabe24ff` |
| `authorizations/development_authorization.json` | `91ec2c5dbea8e4d424abcbb8176df9bfc8e809f46248b8345d41ea530fbbff61` |
| `analyses/development_registered_v1/lr_selection.json` | `dfc9d69faec78cff95e63ed7cd99a0e23c883dad5eba1a2fe378366730e06795` |
| `analyses/development_registered_v1/independent_preconfirmation_review.json` | `6ad8ede4ebdc6517e7d44395ae14dc26cc705029c1efab454eabb366d729803a` |
| `authorizations/confirmation_tuned_authorization.json` | `4d8967d9cb9a499c9cb3f439385d155e3df65fae7c7aed94260b81d1501f71ca` |
| `authorizations/confirmation_common_authorization.json` | `8e09bf3f43f5f6ede5f1b623dfd2d89b7488a7171a87aa9818da53f1f4ec13ab` |
| `analyses/confirmation_registered_v1/confirmation_analysis.json` | `346e46414d82155f2064ee2a448b89cf976bdc6897e0ff8ced06a389056799d6` |
| `analyses/confirmation_registered_v1/common_identity_robustness.json` | `01b2e07de7eaef8e66da938285df90cb9d871b77ffaa9195cfba1fc7ca14b85e` |
| `analyses/confirmation_registered_v1/confirmation_bundle_receipt.json` | `002eba15698ddc91e7360c8f3795cd1dc9562595338658c13bcdcc315b9fb3d6` |

The bundle receipt records 144 complete, zero-failure runs and 37,748,736
paid actions per confirmation directory. On the retained local full payload,
its identity receipt verifies 864 paired scientific binary files across the
two schedules and confirms that all ledgers and five recovery checkpoints per
run are byte-identical. The source lock was local rather than an immutable
public pre-execution commit; that timing limitation remains disclosed.

### Compact-release boundary

The repository does **not** ship the 5,079,086,996-byte full replay payload.
It ships the source/data lock, protocol, tests, authorization and review chain,
the complete 24-block contrast vectors, and
`EXTERNAL_REPLAY_BUNDLE_MANIFEST.json`. That content manifest records the
relative name, byte size, and SHA-256 digest of all 2,904 scientific files
across 420 run directories. The full ledgers and recovery checkpoints remain
on the execution machine, and the manifest's download URI is deliberately
null until they are deposited in content-addressed storage. Consequently, a
clean clone can audit the compact result and rerun the experiment, but cannot
replay or independently verify the historical checkpoints without obtaining
or regenerating that external payload.

## Read-only reproduction

From the repository root, the following checks source/data integrity and the
locked tests without launching any evidentiary development or confirmation
run (some unit tests use one- or two-step temporary engineering fixtures):

```bash
uv run --project curriculum_maxrl/digits_factorial --locked \
  python -m pytest curriculum_maxrl/digits_factorial/tests -q
uv run --project curriculum_maxrl/digits_factorial --locked \
  python -m curriculum_maxrl.digits_factorial.verify_portable \
  --source-only --skip-runtime-check
```

`bash reproduce.sh` additionally checks the receipt chain and manuscript
manifest, including the structure and internal digest of the external-payload
content manifest. It deliberately does not rerun the 288 confirmation runs or
pretend that the unshipped payload is present.
