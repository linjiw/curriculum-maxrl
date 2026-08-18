# What MAZE-SCORE changes about the next study's design

Written 2026-08-18, immediately after the MAZE-SCORE primary landed and before
any next-paper campaign is designed. Outcome-blind with respect to everything
still running (the AMaze confirmatory campaign). Not a claim; a design
constraint I do not want to lose.

## The finding, stated as a constraint

`A_N(p) = 2(1 - p - (1-p)^N)` is exact **for conditionally i.i.d. binary
rollouts**. MAZE-SCORE measured how badly that antecedent fails in a real
neural learner and found the failure is not uniform — it is worst exactly
where a rollout-aware score puts its mass:

| $\hat p$ | realized/predicted mass | silent groups predicted | observed |
|---|---|---|---|
| .11 | .43 | 2.2% | 51.2% |
| .45 | .78 | ~0% | 12.1% |
| .73 | .93 | ~0% | 2.0% |

So there are now **two** distinct gaps between the algebra and learning, not
one:

1. **activity ≠ utility** — the gap the current paper measures (peak location,
   the H=8 audit, the branching pool);
2. **predicted activity ≠ realized activity** — new, and it bites first,
   because it is a failure of the estimator-side prediction itself.

## The design constraint this imposes

**The branching pool cannot see gap 2.** It is synthetic with exact gradients:
rollouts are i.i.d. by construction, so `A_N` is exactly right there and the
realization ratio is identically 1. Every branching result to date — including
the H=20 secondary — was measured in a world where gap 2 does not exist.

That does not invalidate those results. It bounds them: they are statements
about gap 1 *conditional on gap 2 being closed*. If the next paper proposes
`Û_H(x) = A_{E,N}(p_x) + β·r_φ(x,h,G)` and fits `r_φ` only on the branching
pool, `r_φ` will never learn the correction that mattered most at neural
scale.

Three consequences for the next preregistration:

1. **The residual's first job is not structure, it is realization.** Before
   `r_φ` is asked to predict downstream reach, check whether a much simpler
   correction — replacing the binomial `A_N(p̂)` with the empirically realized
   group mass, which is directly observable from telemetry at zero extra cost
   — already recovers most of the loss. If it does, the interesting model is
   `E[mass | task]`, not a graph-structure residual. This is a cheap,
   decisive ablation and it should run first.

2. **The G2 identifiability gate needs a substrate with correlated
   rollouts.** Running CIR on exact-gradient Acrobot measures the stability of
   oracle continuation values in the one regime where the estimator-side
   prediction is exact. The gate should be run where groups share structure
   (a level, a template, a problem family) so that oracle instability from
   over-dispersion is included in the variance it is testing.

3. **Report the realization ratio as standard telemetry.** It costs nothing
   (group `K` is already logged), it is a single number per arm, and it
   distinguishes "the score chose badly" from "the score chose a region the
   estimator could not pay out in". MAZE-SCORE could not have been diagnosed
   without it.

## What this does not license

It does not license reopening MAZE-SCORE, adding seeds, or re-running any arm.
It does not make the over-dispersion diagnosis confirmatory — it is post-hoc
on one task family. And it is not evidence that a realization-corrected score
would win; that is an untested hypothesis, and the honest next step is to test
it prospectively rather than to assert the current paper's negative "would
have" been positive.

Related: [[PI_GUIDANCE_LITPOSITION_2026-08-18]], [[PI_JUDGMENT_2026-08-18]],
`hopper/MAZE_SCORE_RESULT_2026-08-18.md`.
