# Confirmatory matched-pair result (v2) — 2026-08-18

Seeds 5003–5022, warm 200, `|Δp| ≤ .02` **and** `|Δu_N| ≤ .02` absolute,
`C` ratio ≥ 3×. Frozen rule applied once. One MaxRL seed produced no matched
pairs and is dropped (n = 19); RLOO n = 20.

## Primary (H = 8): inconclusive

| estimator | pairs/seed | Δ = U(high-C) − U(low-C) | CI | exact p | pos | verdict |
|---|---|---|---|---|---|---|
| MaxRL | 151 | **+0.00158** | [+.00052, +.00277] | .011 | 11/19 | **inconclusive at n=19** |
| RLOO | 1018 | +0.00004 | [+.00003, +.00006] | <.00001 | 19/20 | activity suffices |

The MaxRL effect is positive and statistically clear (p = .011) but its mean
sits **below** the +0.002 SESOI while its CI spans it. By the frozen rule that
is inconclusive, not support. **v1's exploratory +0.00313 did not reproduce**
at +0.00158 on fresh seeds — roughly half the size. v1 was preregistration-free
and its estimate was optimistic, which is what post-hoc criteria usually buy.

## Frozen secondary: the effect is strongly horizon-dependent

| horizon | MaxRL Δ | exact p | pos | RLOO Δ |
|---|---|---|---|---|
| H = 4 | +0.00096 | .0025 | 13/19 | +0.00003 |
| H = 8 | +0.00158 | .011 | 11/19 | +0.00004 |
| **H = 20** | **+0.00724** | **.00014** | **16/19** | +0.00013 |

Monotone in `H`, and **4.6× larger at H=20 than at H=8**, clearing the SESOI
by more than 3× with the strongest significance in the table. This was a
preregistered secondary, so it is reported as such and **not** promoted to
primary: the honest statement is that the horizon prediction in the PI
judgment's Experiment ② is confirmed in direction and magnitude, and that a
next preregistration should take `H = 20` as its primary.

That is the substantive finding. Two tasks the deployed estimator scores
identically differ in continuation utility by an amount that **grows with how
far into the future you look** — which is precisely what "availability is not
utility" should mean, and precisely what a one-step or short-horizon audit
would miss. The earlier linear-chain audit used H = 8 and found nothing; at
H = 20 on a pool where the factors are separable, the gap is unmistakable.

## The estimator dissociation replicates

Under RLOO the same contrast is +0.00004 at H=8 and +0.00013 at H=20 — 40–55×
smaller than MaxRL at every horizon — even though `ρ(C, U) = +0.874` there,
higher than under MaxRL (+0.689). **Structure predicts utility under both
estimators; it adds utility beyond matched availability only under MaxRL.**
This replicates v1 on fresh seeds and is the cleanest estimator-conditioning
result in the project: the same task-graph fact matters or does not matter
depending on which estimator converts outcomes into updates.

## `A_N·C` remains the wrong form

`ρ(u_N,U) = .677` versus `ρ(u_N·C,U) = .679` under MaxRL — indistinguishable,
as in the linear-chain audit (.641 vs .638) and v1 (.641 vs .638). The
compounding information is real and causal, but multiplying by the structural
count does not express it. This is now three independent measurements agreeing,
and it is the strongest available argument for the judgment's §五 design: a
**learned residual on top of an activity-preserving sampler**, with `β = 0`
recovering the current teacher exactly — not a hand-built `C`.

## Status

Primary inconclusive; horizon secondary confirmed and pointing to the next
preregistration. Nothing here enters the current ICLR submission, whose
boundary is closed. The next step is a v3 preregistration with `H = 20`
primary and a residual predictor as the object under test, per the judgment's
G2 gate — not another matched-pair variant.
