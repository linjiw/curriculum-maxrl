# What MAZE-SCORE changes about the next study's design

> **Superseded in its mechanism wording (2026-08-18).** This note originally
> attributed the effect to within-group over-dispersion. The trainer actually
> repeats ONE concrete maze per group, so rollouts inside a group are
> conditionally i.i.d.; the gap is **task granularity** — the curriculum
> scores a level that aggregates heterogeneous mazes. See
> `PI_CORRECTION_GROUPLAW_GRANULARITY_2026-08-18.md`. The design constraints
> below are unchanged and, if anything, sharper: the branching pool is
> homogeneous by construction, so it cannot exhibit a granularity gap at all.

Written 2026-08-18, immediately after the MAZE-SCORE primary landed and before
any next-paper campaign is designed. Outcome-blind with respect to everything
still running (the AMaze confirmatory campaign). Not a claim; a design
constraint I do not want to lose.

## The finding, stated as a constraint

`A_N(Q) = 2(Pr(K>0) - E[K]/N)` is exact for **any** group law. Its familiar
`p`-only form `2(1 - p - (1-p)^N)` additionally requires that the scored unit
be the unit whose rollouts are i.i.d. MAZE-SCORE measured how badly that
second requirement fails when the curriculum scores a *level* rather than a
maze, and the failure is not uniform — it is worst exactly where a
rollout-aware score puts its mass:

| $\hat p$ | realized/predicted mass | silent groups predicted | observed |
|---|---|---|---|
| .11 | .43 | 2.2% | 51.2% |
| .45 | .78 | ~0% | 12.1% |
| .73 | .93 | ~0% | 2.0% |

So there are now **two** distinct gaps between the algebra and learning, not
one:

1. **activity ≠ utility** — the gap the current paper measures (peak location,
   the H=8 audit, the branching pool);
2. **mean pass rate is not a sufficient statistic for activity** — new, and it
   bites first, because it defeats the estimator-side prediction before any
   question of utility arises. Exactly: the plug-in over-predicts by twice the
   aggregate's excess all-fail probability.

## The design constraint this imposes

**The branching pool cannot see gap 2.** Each pool task is a single concrete
task with exact gradients, so the scored unit *is* the i.i.d. unit, the
granularity gap is identically zero, and the realization ratio is identically
1. Every branching result to date — including
the H=20 secondary — was measured in a world where gap 2 does not exist.

That does not invalidate those results. It bounds them: they are statements
about gap 1 *conditional on gap 2 being closed*. If the next paper proposes
`Û_H(x) = A_{E,N}(p_x) + β·r_φ(x,h,G)` and fits `r_φ` only on the branching
pool, `r_φ` will never learn the correction that mattered most at neural
scale.

Three consequences for the next preregistration:

1. **The residual's first job is not structure, it is the group law.** The
   theorem hands us a directly estimable score with no model at all:
   `û_N(z) = q̂_z − p̂_z`, the observed non-silent fraction minus the observed
   mean pass rate, which is exactly half the realized mass. Test that before
   any learned residual. The minimal deletion ladder is
   (i) i.i.d. plug-in `u_N(p̂_z)`, (ii) group-law `q̂_z − p̂_z`,
   (iii) hierarchical `E_{P∼G_z}[u_N(P)]`, (iv) continuation residual
   `u_N^group(z) + β·r_φ`. Each step repairs exactly one interface, and
   (iv) is only justified if (ii) or (iii) already predicts realized mass and
   still fails to predict continuation utility.

2. **The G2 identifiability gate needs a substrate with a coarse scored
   unit.** Running CIR on exact-gradient Acrobot measures oracle stability in
   the one regime where the scored unit is the i.i.d. unit. The gate should be
   run where the curriculum names an aggregate (a level, template, or problem
   family) so that granularity-induced oracle instability is inside the
   variance it tests.

3. **Report `(q̂ − p̂)` versus `u_N(p̂)` as standard telemetry.** It costs
   nothing (group `K` is already logged), it is two numbers per arm, and their
   difference is exactly the granularity gap. MAZE-SCORE could not have been
   diagnosed without it.

## What this does not license

It does not license reopening MAZE-SCORE, adding seeds, or re-running any arm.
It does not make the granularity diagnosis confirmatory — it is post-hoc on
one task family, and exact accounting for the mass error is not the same as
demonstrated causal mediation of the endpoint. And it is not evidence that a realization-corrected score
would win; that is an untested hypothesis, and the honest next step is to test
it prospectively rather than to assert the current paper's negative "would
have" been positive.

Related: [[PI_GUIDANCE_LITPOSITION_2026-08-18]], [[PI_JUDGMENT_2026-08-18]],
`hopper/MAZE_SCORE_RESULT_2026-08-18.md`.
