# Correlated-rollout stress test for practical MaxRL

## Result in one paragraph

The general coefficient-mass identity survives arbitrary binary rollout
dependence exactly, but the i.i.d. closed form can materially misplace the
curriculum frontier.  For a group with average marginal success probability
`p_bar=E[K]/N`, success count `K`, and practical dropped-group MaxRL
coefficients,

`E[A] = 2(P(K >= 1) - p_bar)`.

Enumerating beta-binomial groups matches this identity to a maximum absolute
error of `2.58e-13`.  Within the beta-binomial family, increasing positive
correlation increases the all-failure probability, so substituting
`1-(1-p)^N` always overstates coefficient activity in the tested grid and
moves the activity-maximizing task toward higher `p`.  Pairwise correlation
alone does not determine an arbitrary binary joint distribution.  This is a
theory-scope result, not evidence that more coefficient mass necessarily
produces better learning.

## Design

- Group sizes: `N={2,4,8,16,32}`.
- Pairwise intra-class correlations:
  `rho={0,.01,.05,.10,.20,.50,.80}`.
- Dependence model: `Q ~ Beta(p c, (1-p)c)`, followed by conditionally i.i.d.
  rollouts given `Q`, where `c=1/rho-1`.  This produces exchangeable binary
  groups with marginal success `p` and pairwise correlation `rho`.
- Peak search: 200,001 points over `p in [1e-6,1-1e-6]`; grid resolution is
  approximately `5e-6`.
- Verification: exact beta-binomial count enumeration at 210 parameter cells,
  plus a fixed-seed 200,000-group Monte Carlo diagnostic for selected cells.

The exact computation is the result.  The Monte Carlo audit is only an
implementation check; its largest deviation from the exact answer is 1.18
estimated standard errors.

## What changes at deployed group sizes

The table reports half coefficient mass, `E[A]/2`, because this is the paper's
sampler utility scale.  “Relative overstatement” compares the i.i.d. value to
the correlated value at the *i.i.d.-optimal* difficulty.

| `N` | `rho` | i.i.d. peak `p` | correlated peak `p` | i.i.d. relative overstatement at its peak |
|---:|---:|---:|---:|---:|
| 16 | .01 | .16876 | .17696 | 1.7% |
| 16 | .05 | .16876 | .20518 | 9.2% |
| 16 | .10 | .16876 | .23420 | 20.0% |
| 16 | .20 | .16876 | .28136 | 45.8% |
| 16 | .50 | .16876 | .38517 | 178.5% |
| 32 | .01 | .10578 | .11691 | 2.1% |
| 32 | .05 | .10578 | .15198 | 12.4% |
| 32 | .10 | .10578 | .18570 | 27.8% |
| 32 | .20 | .10578 | .23908 | 64.3% |
| 32 | .50 | .10578 | .35834 | 247.3% |

Even modest positive correlation matters more as `N` grows.  At the deployed
maze setting `N=32`, `rho=.10` moves the coefficient-activity peak from about
`.106` to `.186`; at `N=16`, it moves from `.169` to `.234`.  The direction is
intuitive: correlated failures make at-least-one-success less likely than the
i.i.d. formula predicts, so the most active task must be easier.

## Paper implication

The safe main-text statement is:

> Under arbitrary binary rollouts, practical MaxRL's expected absolute
> coefficient mass remains `2(P(K>=1)-E[K]/N)`.  Common marginal `p` reduces
> the second term to `p`, and conditional independence supplies
> the closed form `P(K>=1)=1-(1-p)^N`; positive beta-binomial dependence lowers
> activity and shifts its peak toward easier tasks.

This improves the scope discussion without claiming that pairwise correlation
alone fixes activity or that the beta-binomial model describes every policy
or decoding process.  It also suggests a cheap
future diagnostic: estimate within-prompt outcome correlation and replace the
i.i.d. hit probability with an empirical or posterior estimate of
`P(K>=1)`.

## Reproduction

From the repository root:

```bash
python3 -m unittest -v curriculum_maxrl.test_correlated_rollout_stress
python3 curriculum_maxrl/run_correlated_rollout_stress.py
```

The structured artifact is
`curriculum_maxrl/results_correlated_rollout_stress.json`.
