# PI correction — group semantics, and the theorem that replaces the
# "over-dispersion" explanation

**Status:** accepted after direct code audit. The MAZE-SCORE **numbers and
verdict are unchanged**; what changes is the mechanism explanation and, more
importantly, the theory that explains it.

## 1. What I got wrong in 17c4c61

I wrote, in the paper, the result document, the research note, the site and the
README:

> "A group shares a *level*, not a maze, so outcomes inside it are positively
> correlated."

**This is backwards.** `curriculum_maxrl/maze_gpu/train.py` (lines 680–689):

```python
levels = [int(x) for x in teacher.sample_levels(args.tasks_per_step)]
tasks  = [sample_task(lv, rl_task_rngs[lv]) for lv in levels]   # ONE maze per level draw
flat_prompts = [t.prompt for t in tasks for _ in range(args.rollouts)]  # repeated N times
...
teacher.observe(lv, rewards)                                     # posterior pools at the LEVEL
```

The teacher's action is a **level**; one concrete maze is then drawn from that
level; and that single prompt is repeated $N{=}32$ times to form the group. So
**the group shares the maze**, and conditional on that maze the rollouts *are*
i.i.d. The heterogeneity lives *across* groups at a level, and the teacher's
posterior is what pools over it.

So the failure is not "rollouts stopped being i.i.d. at neural scale". It is:

> The theory's task $x$ is a concrete maze. The curriculum's action $z$ is a
> level, which aggregates heterogeneous mazes. Averaging pass rates and then
> applying a nonlinear activity function is not the same as applying activity
> per task and then averaging.

## 2. The identity does not need independence at all

For practical centered drop-all-fail MaxRL with zero stabilizer, the realized
mass of a group with $K$ successes is $M(K)=2(1-K/N)\mathbf 1\{K>0\}$, which is
a deterministic function of $K$. Hence for **any** joint binary group law $Q$ —
no independence, no identical distribution:

$$\mathcal A_{\mathrm{MaxRL},N}(Q)=\mathbb E_Q[M]=2\Big(\Pr_Q(K>0)-\tfrac{\mathbb E_Q[K]}{N}\Big)=2(q_N-\bar p).$$

The familiar $A_N(p)=2\{1-p-(1-p)^N\}$ is the **conditional-i.i.d. slice**,
obtained by $q_N=1-(1-p)^N$. So:

> The group-law coefficient-mass identity is exact without independence; what
> requires conditional i.i.d. rollouts is the scalar $p$-only reduction, and
> with it the factorization $p(1-p)w_{N-1}(p)$ and the peak $p^\star_N$.

Estimator activity still *is* realized activity, exactly. What fails is
predicting it from a coarse mean pass rate.

## 3. The task-granularity gap, exactly

Let the curriculum score aggregate $z$, with $X\sim\nu(\cdot\mid z)$ the
concrete task and rollouts conditionally i.i.d. given $X$. Then

$$A_N^{\text{true}}(z)=2\,\mathbb E_X[u_N(p_X)],\qquad A_N^{\text{plug}}(z)=2\,u_N(\bar p_z).$$

$u_N''(p)=-N(N-1)(1-p)^{N-2}<0$, so $u_N$ is strictly concave and Jensen gives
$A^{\text{true}}\le A^{\text{plug}}$: **mean-pass-rate plug-in always
over-predicts activity on a heterogeneous aggregate.** The gap is exact:

$$A_N^{\text{plug}}(z)-A_N^{\text{true}}(z)=2\Big[\Pr(K=0\mid z)-(1-\bar p_z)^N\Big],$$

i.e. **twice the aggregate's excess all-fail probability** over the binomial at
the same mean.

**Verified to floating point on the campaign**
(`curriculum_maxrl/maze_score/group_law_audit.py`,
`hopper/MAZE_SCORE_GROUPLAW_AUDIT.json`): over 41,101 / 18,497 / 9,355
(seed, arm, level, window) cells at window widths 10 / 25 / 50 updates, the
maximum absolute deviation is $2.8\times10^{-16}$ for
$\bar M=2(\hat q-\hat p)$ and $4.4\times10^{-16}$ for the excess-silence
identity. This is algebraic accounting, not a fitted explanation.

## 4. Why $u_{32}$ is hurt more than $p(1-p)$

Second order, $u_N(\bar p)-\mathbb E[u_N(P)]\approx\tfrac12 N(N-1)(1-\bar
p)^{N-2}\operatorname{Var}(P)$. At $u_N$'s own peak
$|u_N''(p^\star_N)|=(N-1)/(1-p^\star_N)$, which is $\approx 34.7$ at $N=32$, a
local penalty coefficient $\approx 17.3\operatorname{Var}(P)$. For the
canonical score the relation is exact and much milder:
$u_2(\bar p)-\mathbb E[u_2(P)]=\operatorname{Var}(P)$.

So there is a previously unstated trade-off:

> Raising $N$ moves estimator-side activity toward harder tasks **and** makes
> that geometry more sensitive to curriculum granularity. Rollout awareness and
> task heterogeneity pull against each other.

Seed-clustered over the 48 blocks (never over the 288,000 group draws): the
realization ratio is $.580$ $[.570,.590]$ for $u_{32}$ and $.703$
$[.691,.715]$ for $p(1-p)$; paired, $-.1227$ $[-.1332,-.1117]$, negative in
**48/48** blocks.

## 5. Claim boundaries this correction imposes

- The MAZE-SCORE primary, secondary, SESOI and verdict are **unchanged**. This
  authorizes **no** re-run, no added seed, no endpoint substitution.
- The granularity gap exactly accounts for the **coefficient-mass prediction
  error**. It has *not* been shown by intervention to be the sole causal
  mediator of the endpoint contrast. It is consistent with it; that is the
  claim.
- "Neural scale" is a property of this protocol, not an isolated causal
  variable: the study varies model capacity, function approximation, procedural
  task heterogeneity and curriculum granularity together.
- Post-hoc telemetry pools maze-to-maze heterogeneity with learner
  nonstationarity inside a window; the honest description is *latent
  task-and-state heterogeneity*. The identity check is window-invariant, which
  bounds but does not eliminate the concern.
- No counterfactual "a corrected score would have won" is asserted anywhere.

## 6. Also found in the audit

`maze_env.sample_task`'s docstring says "Fresh 13x13 maze"; `SIZE = 17` and
`LEVEL_DIST` runs 4…28 in steps of 2. The mazes are **17×17 across 13
goal-distance levels**, as the manuscript now says. The docstring is stale;
the source is *not* edited here because its hash is bound into the frozen
campaign bundle.

Related: [[PI_GUIDANCE_LITPOSITION_2026-08-18]],
`RESEARCH_NOTE_GRANULARITY_2026-08-18.md` (superseded in its mechanism
wording by this note), `hopper/MAZE_SCORE_RESULT_2026-08-18.md`.
