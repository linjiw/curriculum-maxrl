# Acrobot V3: post-hoc mechanism and native-task audit

Status: **complete, historical/post hoc, no new training**.

This audit reads the 40 frozen Acrobot V3 runs (20 paired seeds, two arms,
2,000,000 nominal transitions per run) and derives quantities that were not
part of the original registered hypothesis test. The retained target-uniform
threshold-family AUC contrast is `u16 - uniform = +0.03635`, but a later audit
found cross-seed RNG-domain reuse: action root `s+1` for logical seed `s`
equals the parameter-initialization root of logical seed `s+1`. We therefore
do not retain the original paired interval or sign-flip test as clean
confirmatory inference.

As a descriptive sensitivity check, non-neighboring even seeds give mean
`+0.02512` (6/10 positive; sign-flip `p=.09961`) and odd seeds give mean
`+0.04759` (8/10 positive; `p=.01563`). Both means remain positive, but the
split is imprecise and does not repair the original design.

## What the no-training audit adds

For an observed practical-MaxRL group with `K` successes among `N=16`
rollouts, the realized absolute coefficient mass is zero at `K=0` and `K=16`,
and otherwise `2(16-K)/16`. Recomputing this from every saved group gives:

| Frozen paired outcome | Uniform | u16 curriculum | Paired difference | Pair signs |
|---|---:|---:|---:|---:|
| coefficient mass / sampled group | 0.6179 | 0.7510 | **+0.1331** | 20+/0-/0= |
| coefficient mass / million transitions | 105.40 | 116.19 | **+10.79** | 17+/3-/0= |
| nonzero-mass group fraction | 0.7507 | 0.8789 | **+0.1282** | 20+/0-/0= |
| optimizer updates / million transitions | 128.72 | 136.47 | **+7.75** | 20+/0-/0= |
| native Acrobot success AUC | 0.3122 | 0.3763 | **+0.0641** | 15+/5-/0= |
| native Acrobot return AUC | -465.67 | -457.40 | **+8.26** | 16+/4-/0= |
| final native success rate | 0.6203 | 0.7469 | **+0.1266** | 18+/1-/1= |

The teacher also produced 15.55 more live updates per run and 47.4 fewer
all-pass groups on average. Its episodes were longer (364.3 versus 402.6
transitions per episode), which is why matching and normalizing by paid
transitions is essential. Descriptively, on a standard Gymnasium dynamics
task, the exact `u16` sampler increased the amount of practical-MaxRL
coefficient activity purchased by both a group and a paid transition, while
the same runs improved native success and native return.

## Claim boundary

These endpoints were selected after the V3 outcomes existed. Their bootstrap
intervals and sign-flip tests in the JSON are descriptive and unadjusted; the
RNG-domain finding additionally prevents clean paired inference for the
historical primary. They do not prove coefficient mass is a causal mediator, and they
do not compare against `p(1-p)`, PLR, ALP, PAIRED, or ACCEL. The strongest
fresh CPU follow-up is a source-frozen Acrobot tournament with uniform,
`p(1-p)`, and `u16` under the same practical-MaxRL learner and transition
budget; PLR and ALP can be added as secondary historical-dynamics baselines.
