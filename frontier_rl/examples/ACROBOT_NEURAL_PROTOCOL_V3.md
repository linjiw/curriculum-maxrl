# Neural Acrobot validation protocol — V3 shared-policy efficacy confirmation

Status: frozen on 2026-07-22 after inspecting the excluded V1 pilot and V2
capacity-development results, and before running any V3 observation on the
sealed confirmatory seeds `12000..12019`.

This document narrows the claim authorized by
[`ACROBOT_NEURAL_PROTOCOL_V2.md`](ACROBOT_NEURAL_PROTOCOL_V2.md). It does not
rewrite or erase either earlier protocol, their retained artifacts, or their
failed launch gates.

## 1. Why the V2 causal matrix stopped

The excluded V2 six-cell development artifact is
`acrobot_neural_v2_capacity_development.json`, SHA-256
`915c2e476560b02f8f8017e1d97984c39c27f3ad1b7743e52651f8f14acb95bd`.
Its gate record is `acrobot_neural_v2_development_gates.json`, SHA-256
`bd888dead27d9ac577ac85507b008206bf311de480abc09eb79182b2add51fe7`.

Five of six effect-blind launch gates passed: source and numerical invariants,
mixed-regime exposure, nonuniform teacher behavior, observation of every
success-count regime, and projected runtime. The every-cell learning/headroom
gate failed. At the first complete-group checkpoint at or above one million
transitions, the numbers of qualifying seeds were `3/3` and `3/3` for shared
uniform and shared teacher, but `0/3`, `0/3`, `0/3`, and `2/3` for the two
disjoint controls. Therefore V2 authorizes no six-cell confirmation and no
causal transfer or capacity-control claim.

Only after that gate decision was fixed, the excluded shared-policy development
contrast was inspected. Teacher-minus-uniform transition AUC was approximately
`+0.0392` over three paired seeds. This is an exploratory rationale for a new,
narrower confirmation; it is not included in the confirmatory estimate or
test.

## 2. Claim and non-claim

V3 asks one question:

> With one task-agnostic shared H64 neural policy and no hindsight relabeling,
> does exact frontier curriculum sampling improve target-uniform learning
> efficiency over uniform task sampling on the fixed Acrobot threshold family?

The sole confirmatory claim is **shared-policy curriculum efficacy**. V3 does
not evaluate whether improvement is caused by cross-task transfer, parameter
sharing, task relatedness, a capacity advantage, or hindsight. The failed V2
controls remain visible and cannot be rehabilitated by a positive V3 result.

## 3. Frozen environment and algorithm

- Environment: official Gymnasium `Acrobot-v1`, with its 500-step time limit.
- Fixed ordered task thresholds:
  `[-1.5, -1.0, -0.5, 0.0, 0.25, 0.5, 0.7, 1.0]`.
- Verifier: binary success when the post-transition Acrobot tip height is
  strictly greater than the requested threshold.
- Actor: one task-agnostic shared categorical policy, one tanh hidden layer of
  width 64, 640 trainable parameters, no output bias.
- Update: frozen-policy SUM score-function gradient and plain SGD ascent.
- Group size: `N=16`; every group is completed even when the nominal transition
  budget is crossed.
- Uniform arm: each requested task is sampled with probability `1/8`.
- Curriculum arm: exact frontier utility
  `u(p)=1-(1-p)^16-p`, normalized after applying teacher decay `0.7` and floor
  `0.1`, with `gamma=1`.
- Hindsight scale: zero in both arms. No hindsight-derived update is applied.
- Learning rate: `3e-4`, selected by V2's effect-blind within-arm
  learning/headroom rule.

All implementation details not amended here remain as frozen in V1 and V2.

## 4. Sealed design

Run exactly two paired cells:

```text
US: uniform sampling, shared H64
CS: exact frontier teacher, shared H64
```

Use exactly 20 paired seeds, `12000..12019`. A new high block is used because
seed `0` appears in the earlier core smoke and several low seeds appear in
adapter tests. Repository Acrobot artifacts and tests were checked before
freezing V3; none had executed the `12000..12019` block.

Each cell receives a nominal budget of 2,000,000 actual environment
transitions. The final 16-rollout group is completed, so the recorded terminal
coordinate can exceed that nominal budget. Evaluate at initialization and
after the first completed group crossing each 100,000-transition boundary,
using 32 fresh episodes per target task and fixed per-seed evaluation common
random numbers. Evaluation must not mutate training parameters, counters, or
random-generator states.

There is no interim analysis, outcome-dependent stopping, seed replacement,
hyperparameter change, or pooling with V1/V2 data. Every run is retained. If
any of the 40 runs is missing, non-finite, or fails a registered accounting,
verifier, parameter-count, evaluation-state, or source-lock check, the primary
analysis is not performed.

## 5. Primary estimand and decision

For seed `s` and method `m`, let `q_{s,m}(x)` be the target-uniform mean pass
rate recorded against actual environment transitions. Let `A_{s,m}` be its
normalized trapezoidal area from the initialization coordinate through that
run's completed terminal coordinate. The paired effect is

```text
d_s = A_s,CS - A_s,US,
Delta_hat = (1/20) sum_s d_s.
```

The only hypothesis test is the exact two-sided paired sign-flip randomization
test over all `2^20` sign assignments:

```text
p = 2^-20 sum_e 1[ |mean_s(e_s d_s)| >= |Delta_hat| ].
```

Its randomization interpretation assumes paired effects are sign-exchangeable
under the sharp null. Pairing removes seed-level initialization and evaluation
noise shared by the two arms; it does not make this assumption automatic.

The shared-policy efficacy claim is supported if and only if both conditions
hold:

1. `Delta_hat >= +0.03` normalized transition-AUC units; and
2. the exact two-sided sign-flip `p <= 0.05`.

Because there is one predeclared test, there is no multiplicity adjustment.
A 20,000-resample paired-seed bootstrap 95% interval is descriptive support.
Failure is reported as not confirmed, never as equivalence or evidence of no
effect.

Final mean pass, hardest-task pass, native Acrobot success, native return,
censored time to goal, transitions, updates, and throughput are secondary
descriptive outcomes. They cannot rescue a failed primary decision.

## 6. Reproducibility and claim boundary

Before execution, `ACROBOT_NEURAL_V3_LOCK.json` must freeze SHA-256 hashes of
the runner, adapter, teacher, estimators, interfaces, this protocol, and the
independent V3 verifier, together with Python, NumPy, and Gymnasium versions.
The confirmatory artifact must reproduce those hashes exactly. The verifier
must recompute every per-run AUC from saved curves, the paired contrast, exact
test, and decision without changing the artifact.

A positive V3 result supports only the question in Section 2 on this fixed
Acrobot threshold family. Generalization beyond Acrobot, transfer causality,
and hindsight-scale benefits require separate, versioned studies.
